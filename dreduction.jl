using MPI
using ChainRulesCore

function ChainRulesCore.rrule(
    ::typeof(MPI.Allreduce),
    buf::Float64,
    op::MPI.Op,
    comm::MPI.Comm,
)
    ret = MPI.Allreduce(buf, op, comm)
    function dAllreduce(dbuf)
        dret = MPI.Allreduce(dbuf, op, comm)
        np = MPI.Comm_size(comm)
        return NoTangent(), dret/float(np),
               NoTangent(), NoTangent()
    end
    return ret, dAllreduce
end
