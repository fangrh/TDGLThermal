"""Pure checks for continuation receipt hash metadata; no filesystem or RNG I/O."""

const _HEX64 = r"^[0-9a-f]{64}$"

function checked_preceding_sha(preceding_path, actual_child_initial_source_hash,
                               expected_previous_output_hash)
    if preceding_path === nothing
        actual_child_initial_source_hash === nothing ||
            error("fresh segment unexpectedly has initial_source_hash")
        expected_previous_output_hash === nothing ||
            error("fresh segment unexpectedly has previous output hash")
        return nothing
    end
    actual_child_initial_source_hash isa AbstractString ||
        error("continuation missing child initial_source_hash")
    expected_previous_output_hash isa AbstractString ||
        error("continuation missing expected previous output hash")
    occursin(_HEX64, actual_child_initial_source_hash) ||
        error("child initial_source_hash must be lowercase SHA-256")
    occursin(_HEX64, expected_previous_output_hash) ||
        error("expected previous output hash must be lowercase SHA-256")
    actual_child_initial_source_hash == expected_previous_output_hash ||
        error("child initial_source_hash does not match previous output hash")
    return actual_child_initial_source_hash
end

function checked_output_sha(output_hash)
    output_hash isa AbstractString && occursin(_HEX64, output_hash) ||
        error("manifest output_hash must be lowercase SHA-256")
    output_hash
end
