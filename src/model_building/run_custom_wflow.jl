using Wflow

using UnPack
using ProgressLogging
using NCDatasets
using Interpolations
using Base.Threads
using Statistics

function run_custom_indexing(model::Wflow.Model; close_files=true)
    @unpack network, config, writer, clock = model

    # in the case of sbm_gwf it's currently a bit hard to use dispatch
    model_type = config.model.type::String

    # determine timesteps to run
    calendar = get(config, "calendar", "standard")::String
    starttime = clock.time
    Δt = clock.Δt
    endtime = Wflow.cftime(config.endtime, calendar)
    times = range(starttime, endtime, step=Δt)

    # get forcing settings
    lapse_correction = get(config.forcing, "lapse_correction", false)::Bool
    @info "Lapse rate correct is set to $lapse_correction"
    if lapse_correction
        # read lapse rate settings and load layers
        lapse_rate = get(config.forcing, "lapse_rate", -0.0065)::Float64
        path_orography = config.forcing.path_orography
        abspath_orography = Wflow.input_path(config, path_orography)
        forcing_name = get(config.forcing, "layer_name", "wflow_dem")::String
        @info "Reading forcing dem from `$abspath_orography` with layer `$forcing_name`"
        orography = NCDataset(abspath_orography)[forcing_name]
        # calculate temperature correction map for the forcing_name
        correct2sea = temp_correction(orography, lapse_rate)

        wflow_name = get(config.input.vertical, "altitude", "wflow_dem")::String
        @info "Reading Wflow DEM from based on layer `$wflow_name` in the staticmaps"
        wflow_dem = Wflow.read_standardized(model.reader.cyclic_dataset, wflow_name, (x=:, y=:))
        # calculate temperature correction map for the model elevation
        correct2dem = temp_correction(wflow_dem, lapse_rate)
    else
        correct2sea = nothing
        correct2dem = nothing
    end

    # get index mappings
    path_idx = Wflow.input_path(config, get(config.forcing, "path_idx", ""))
    ds_idx = NCDataset(path_idx)
    lon_idx = Wflow.read_standardized(ds_idx, get(config.forcing, "lon_idx_name", "lon_idx"), (x=:, y=:))
    lat_idx = Wflow.read_standardized(ds_idx, get(config.forcing, "lat_idx_name", "lat_idx"), (x=:, y=:))
    indices = LinearIndices(size(lat_idx))

    @info "Run information" model_type starttime Δt endtime nthreads()
    runstart_time = now()
    @progress for (i, time) in enumerate(times)
        @debug "Starting timestep" time i now()
        load_dynamic_input_custom!(model, correct2sea, correct2dem, lon_idx, lat_idx, indices)
        model = Wflow.update(model)
    end
    @info "Simulation duration: $(canonicalize(now() - runstart_time))"

    # write output state NetCDF
    # undo the clock advance at the end of the last iteration, since there won't
    # be a next step, and then the output state falls on the correct time
    Wflow.rewind!(clock)
    Wflow.write_netcdf_timestep(model, writer.state_dataset, writer.state_parameters)

    Wflow.reset_clock!(model.clock, config)

    # option to support running function twice without re-initializing
    # and thus opening the NetCDF files
    if close_files
        Wflow.close_files(model, delete_output=false)
    end
    return model
end

"Get dynamic and cyclic NetCDF input"
function load_dynamic_input_custom!(model, correct2sea, correct2dem, lon_idx, lat_idx, indices)

    update_forcing_indexing!(model, correct2sea, correct2dem, lon_idx, lat_idx, indices)
    if haskey(model.config.input, "cyclic")
        Wflow.update_cyclic!(model)
    end
end

"Get dynamic NetCDF input for the given time"
function update_forcing_indexing!(model, correct2sea, correct2dem, lon_idx, lat_idx, indices)
    @unpack vertical, clock, reader, network, config = model
    @unpack dataset, dataset_times, forcing_parameters = reader

    do_reservoirs = get(config.model, "reservoirs", false)::Bool
    do_lakes = get(config.model, "lakes", false)::Bool

    if do_reservoirs
        sel_reservoirs = network.reservoir.indices_coverage
        param_res = Wflow.get_param_res(model)
    end
    if do_lakes
        sel_lakes = network.lake.indices_coverage
        param_lake = Wflow.get_param_lake(model)
    end

    # get forcing settings
    lapse_correction = get(config.forcing, "lapse_correction", false)::Bool

    # print(lapse_correction, lapse_rate, orography)

    # Wflow expects `right` labeling of the forcing time interval, e.g. daily precipitation
    # at 01-02-2000 00:00:00 is the accumulated total precipitation between 01-01-2000
    # 00:00:00 and 01-02-2000 00:00:00.

    # load from NetCDF into the model according to the mapping
    for (par, ncvar) in forcing_parameters
        # no need to update fixed values
        ncvar.name === nothing && continue

        time = convert(eltype(dataset_times), clock.time)

        t_index = findfirst(>=(time), dataset_times)
        time < first(dataset_times) && throw(DomainError("time $time before dataset begin $(first(dataset_times))"))
        t_index === nothing && throw(DomainError("time $time after dataset end $(last(dataset_times))"))

        # data = Wflow.get_at(dataset, ncvar.name, dataset_times, time)
        data = dataset[ncvar.name][:, :, t_index]

        if ncvar.scale != 1.0 || ncvar.offset != 0.0
            data .= data .* ncvar.scale .+ ncvar.offset
        end

        if par[2] == :temperature
            if lapse_correction

                data = data - correct2sea
                data = forcing_to_wflow_grid(data, lon_idx, lat_idx, indices)
                data = data + correct2dem

            else
                data = forcing_to_wflow_grid(data, lon_idx, lat_idx, indices)
            end
        else
            data = forcing_to_wflow_grid(data, lon_idx, lat_idx, indices)
        end

        # calculate the mean precipitation and evaporation over the lakes and reservoirs
        # and put these into the lakes and reservoirs structs
        # and set the precipitation and evaporation to 0 in the vertical model
        if par in Wflow.mover_params
            if do_reservoirs
                for (i, sel_reservoir) in enumerate(sel_reservoirs)
                    avg = mean(data[sel_reservoir])
                    data[sel_reservoir] .= 0
                    param_res[par][i] = avg
                end
            end
            if do_lakes
                for (i, sel_lake) in enumerate(sel_lakes)
                    avg = mean(data[sel_lake])
                    data[sel_lake] .= 0
                    param_lake[par][i] = avg
                end
            end
        end

        param_vector = Wflow.param(model, par)
        sel = Wflow.active_indices(network, par)
        data_sel = data[sel]
        if any(ismissing, data_sel)
            print(par)
            msg = "Forcing data has missing values on active model cells for $(ncvar.name)"
            throw(ArgumentError(msg))
        end
        param_vector .= data_sel
    end

    return model
end



function forcing_to_wflow_grid(raw_data, lon_idx, lat_idx, indices)
    # Create a temperorary array to store the forcing values
    result = similar(indices, eltype(raw_data))
    # Fill the data array with the corresponding values from the raw data
    for i in eachindex(indices)
        result[i] = raw_data[lon_idx[indices[i]], lat_idx[indices[i]]]
    end
    # Reshape array into a 2D array similar to the wflow grid
    result_2d = reshape(result, size(lat_idx))
    return result_2d
end



function temp_correction(dem, lapse_rate)
    return dem * lapse_rate
end


using Dates
using Formatting


# Read the TOML and initialize SBM model
# tomlpath = "data/wflow_settings_base.toml"

# # Get the TOML path from CLI
# n = length(ARGS)
# if n != 1
#     throw(ArgumentError(usage))
# end
# tomlpath = only(ARGS)
# if !isfile(tomlpath)
#     throw(ArgumentError("File not found: $(tomlpath)\n"))
# end

function run(tomlpath::AbstractString; silent=nothing)

    config = Wflow.Config(tomlpath)

    # if the silent kwarg is not set, check if it is set in the TOML
    silent = get(config, "silent", false)::Bool
    fews_run = get(config, "fews_run", false)::Bool
    logger, logfile = Wflow.init_logger(config; silent)
    Wflow.with_logger(logger) do
        try
            model = Wflow.initialize_sbm_model(config)
            model = run_custom_indexing(model)
        catch e
            # avoid logging backtrace for the single line FEWS log format
            # that logger also uses SimpleLogger which doesn't result in a good backtrace
            if fews_run
                @error "Wflow simulation failed" exception = e _id = :wflow_run
            else
                @error "Wflow simulation failed" exception = (e, catch_backtrace()) _id =
                    :wflow_run
            end
            rethrow()
        finally
            close(logfile)
        end
    end
end

run(ARGS[1])
## julia --project="C:\Users\buitink\.julia\environments\across" -t 4 run_custom_wflow.jl "data/wflow_settings_base.toml"

