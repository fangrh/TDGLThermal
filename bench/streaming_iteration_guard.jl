"""Checked stepping for the streaming harness.

The PDE/model source is intentionally unchanged.  This helper only replaces the
benchmark harness' target-advance loop so a solver return code or a stalled time
coordinate cannot become a busy loop.  `TDGL_STREAM_MAXITERS` is an optional,
explicit harness override; when unset the solver's native default is preserved.
"""

using SHA

function stream_maxiters_override()
    raw = get(ENV, "TDGL_STREAM_MAXITERS", "")
    isempty(raw) && return nothing
    occursin(r"^[1-9][0-9]*$", raw) ||
        error("TDGL_STREAM_MAXITERS must be a positive integer, got '$raw'")
    value = try
        parse(Int, raw)
    catch
        error("TDGL_STREAM_MAXITERS is outside the supported Int range: '$raw'")
    end
    value > 0 || error("TDGL_STREAM_MAXITERS must be positive")
    return value
end

const STREAM_GUARD_SOURCE_HASH = bytes2hex(SHA.sha256(read(@__FILE__)))

function checked_stream_step!(integrator)
    before = integrator.t
    rc = step!(integrator)
    rc in (ReturnCode.Success, ReturnCode.Default) ||
        error("stream integrator stopped at t=$(integrator.t), retcode=$rc")
    integrator.t > before ||
        error("stream integrator made no time advancement at t=$(integrator.t), retcode=$rc")
    return rc
end

"""Harness-only replacement matching the original add-tstop!/step! sequence."""
function advance_integrator_to!(integrator, target_t::Float64)
    target_t <= integrator.t && return
    add_tstop!(integrator, target_t)
    override = stream_maxiters_override()
    override === nothing || (integrator.opts.maxiters = override)
    while integrator.t < target_t
        checked_stream_step!(integrator)
    end
    integrator.t >= target_t ||
        error("Integrator failed to reach target time $target_t (stopped at $(integrator.t))")
    return nothing
end
