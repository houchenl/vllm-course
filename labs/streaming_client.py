"""Minimal local completion-stream observer. Times chunks, not individual tokens."""

import json
import time
import traceback
import urllib.error
import urllib.request
import uuid


def token_workload(tokenizer_path,length,variant=0):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    seed=tokenizer.encode(f"Synthetic course workload variant {variant}. ",add_special_tokens=False)
    if not seed:
        raise ValueError("empty token seed")
    return (seed*(length//len(seed)+1))[:length]


def observe(base_url,model,prompt,output_length=32,timeout=60):
    request_id="course-"+str(uuid.uuid4())
    body=dict(model=model,prompt=prompt,max_tokens=output_length,min_tokens=output_length,
              ignore_eos=True,temperature=0,seed=17,stream=True,
              stream_options={"include_usage":True})
    request=urllib.request.Request(base_url.rstrip("/")+"/v1/completions",
                                   data=json.dumps(body).encode(),
                                   headers={"Content-Type":"application/json","X-Request-Id":request_id})
    started=time.monotonic()
    events=[]
    usage=None
    finish_reason=None
    done=False
    error=None
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            data_lines=[]
            for raw in response:
                line=raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line or not data_lines:
                    continue
                data="\n".join(data_lines)
                data_lines=[]
                if data=="[DONE]":
                    done=True
                    break
                message=json.loads(data)
                if message.get("error"):
                    raise RuntimeError(str(message["error"]))
                if message.get("usage"):
                    usage=message["usage"]
                for choice in message.get("choices",[]):
                    if choice.get("index",0)!=0:
                        raise RuntimeError("unexpected completion choice index")
                    text=choice.get("text","")
                    if text:
                        events.append(dict(seconds=time.monotonic()-started,text=text))
                    if choice.get("finish_reason") is not None:
                        finish_reason=choice["finish_reason"]
            if data_lines and not done:
                raise RuntimeError("stream ended with an incomplete SSE event")
        if not done or not events or finish_reason is None:
            raise RuntimeError("completion stream incomplete or produced no nonempty text")
        if usage is not None:
            if usage.get("prompt_tokens")!=len(prompt):
                raise RuntimeError("server input token count differs from token-ID workload")
            if usage.get("completion_tokens")!=output_length:
                raise RuntimeError("server output length differs from fixed workload")
    except Exception as exc:
        if isinstance(exc,urllib.error.HTTPError):
            error=dict(type=type(exc).__name__,message=str(exc),body=exc.read().decode(errors="replace"))
        else:
            error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finished=time.monotonic()
    return dict(request_id=request_id,success=error is None,error=error,
                started_monotonic=started,finished_monotonic=finished,
                prompt_token_ids=prompt,requested_output_tokens=output_length,
                usage=usage,counts_verified=usage is not None and error is None,
                first_nonempty_chunk_seconds=events[0]["seconds"] if events else None,
                end_to_end_seconds=finished-started,
                nonempty_chunk_events=events,finish_reason=finish_reason,
                measurement="client first nonempty text chunk and completion; no per-token ITL or device timing")
