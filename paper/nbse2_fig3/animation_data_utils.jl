using Statistics

function trim_snapshot_data_for_storage(times, V, psi, T, skip_ratio::Float64, extras...)
    n_snap = minimum((
        length(times),
        length(V),
        size(psi, 3),
        size(T, 3),
        map(extra -> size(extra, 3), extras)...,
    ))

    skip_idx = max(1, Int(ceil(n_snap * skip_ratio)))
    trimmed_extras = map(extra -> extra[:, :, skip_idx:n_snap], extras)

    return (
        original_n_snap = n_snap,
        skip_idx = skip_idx,
        times = times[skip_idx:n_snap],
        V = V[skip_idx:n_snap],
        psi = psi[:, :, skip_idx:n_snap],
        T = T[:, :, skip_idx:n_snap],
        extras = trimmed_extras,
    )
end

function normalize_animation_snapshot_data(times, V, psi, T, skip_ratio::Float64; already_trimmed::Bool=false)
    n_snap = minimum((
        length(times),
        length(V),
        size(psi, 3),
        size(T, 3),
    ))

    times_aligned = times[1:n_snap]
    V_aligned = V[1:n_snap]
    psi_aligned = psi[:, :, 1:n_snap]
    T_aligned = T[:, :, 1:n_snap]

    skip_idx = already_trimmed ? 1 : max(1, Int(ceil(n_snap * skip_ratio)))
    T_avg = vec(mean(T_aligned[2:end-1, 2:end-1, :], dims=(1, 2)))
    V_avg = mean(V_aligned[skip_idx:end])
    V_t = [(times_aligned[i], V_aligned[i]) for i in skip_idx:n_snap]

    return (
        times = times_aligned,
        V = V_aligned,
        psi = psi_aligned,
        T = T_aligned,
        T_avg = T_avg,
        skip_idx = skip_idx,
        V_avg = V_avg,
        V_t = V_t,
    )
end
