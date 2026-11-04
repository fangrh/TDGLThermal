# Independent encoded-frame verification using the already installed Julia
# FFmpeg artifact, including hosts without ffprobe on PATH.
using Plots,JSON3,SHA,Dates
length(ARGS)==2 || error("usage: verify_rendered_video.jl render_receipt.json output.json")
receipt_path,out=ARGS
r=JSON3.read(read(receipt_path,String))
fm=only([m for (id,m) in Base.loaded_modules if id.name=="FFMPEG"])
rows=[]
for run in r.runs
    path=String(run.actual_movie_path)
    lines=fm.exe("-v","error","-select_streams","v:0","-count_frames",
        "-show_entries","stream=nb_read_frames,width,height,r_frame_rate","-of","json",path;
        command=fm.ffprobe,collect=true)
    data=JSON3.read(join(lines,"\n"));s=only(data.streams)
    frames=parse(Int,String(s.nb_read_frames))
    frames==Int(run.expected_frames) || error("encoded frame count differs from requested frames")
    Int(s.width)==1400 && Int(s.height)==900 || error("unexpected movie dimensions")
    String(s.r_frame_rate)=="$(Int(r.fps))/1" || error("unexpected encoded frame rate")
    push!(rows,(backend=String(run.backend),repetition=Int(run.repeat),movie_path=path,
        sha256=bytes2hex(SHA.sha256(read(path))),frames=frames,width=Int(s.width),height=Int(s.height),fps=String(s.r_frame_rate)))
end
length(rows)==6 || error("expected six measured movies")
result=(schema="tdgl-rendered-video-verification-v1",valid=true,created_utc=string(now(UTC)),
    input_receipt_hash=bytes2hex(SHA.sha256(read(receipt_path))),
    tool="FFMPEG.jl bundled ffprobe",all_movie_hashes_equal=length(unique(x.sha256 for x in rows))==1,rows=rows)
open(out,"w") do io;write(io,JSON3.write(result));end
println(JSON3.write((valid=true,movies=length(rows),all_movie_hashes_equal=result.all_movie_hashes_equal)))
