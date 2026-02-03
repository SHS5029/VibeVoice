#!/usr/bin/env python
"""VibeVoice ASR Microphone Test CLI"""
import sys, os, argparse, time as time_mod, numpy as np, queue

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: pip install sounddevice"); sys.exit(1)
try:
    import soundfile as sf
except ImportError:
    print("ERROR: pip install soundfile"); sys.exit(1)
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else None
except ImportError:
    print("ERROR: pip install torch"); sys.exit(1)

try:
    from demo.vibevoice_asr_gradio_demo import VibeVoiceASRInference
except ImportError as e:
    print(f"ERROR: {e}\ncd VibeVoice 폴더에서 실행하세요"); sys.exit(1)

try:
    from vibevoice.llm_client import LLMClient
except ImportError as e:
    print(f"WARNING: LLMClient not available: {e}")

def list_devices():
    print("\n=== 오디오 장치 ===")
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            default = " (기본)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{default}")

def record(duration=5, sr=16000, device=None):
    q = queue.Queue()
    def cb(data, frames, t, status):
        if status: print(f"경고: {status}", file=sys.stderr)
        q.put(data.copy())
    
    print(f"\n녹음 {duration}초 - 말씀하세요!")
    try:
        with sd.InputStream(samplerate=sr, channels=1, callback=cb, device=device, dtype='float32'):
            start = time_mod.time()
            while time_mod.time() - start < duration:
                r = duration - (time_mod.time() - start)
                p = int(20 * (1 - r/duration))
                print(f"\r[{'='*p}{'-'*(20-p)}] {int(r)+1}초  ", end="", flush=True)
                time_mod.sleep(0.1)
        print(f"\r[{'='*20}] 완료!          ")
    except sd.PortAudioError as e:
        print(f"\n마이크 오류: {e}\n--list-devices로 확인하세요"); sys.exit(1)
    
    chunks = []
    while not q.empty(): chunks.append(q.get())
    return np.concatenate(chunks) if chunks else None

def print_segments(segments):
    """Print transcription segments to console."""
    if not segments:
        print("\n📋 Audio Segments: None available")
        return
    print(f"\n📋 Audio Segments ({len(segments)} segments):")
    print("=" * 60)
    for seg in segments[:50]:  # Show first 50
        start = seg.get('start_time', 'N/A')
        end = seg.get('end_time', 'N/A')
        speaker = seg.get('speaker_id', 'N/A')
        text = seg.get('text', '')
        print(f"[{start} - {end}] Speaker {speaker}: {text}")
    if len(segments) > 50:
        print(f"  ... and {len(segments) - 50} more segments")
    print("=" * 60)

def print_analysis(analysis):
    """Print LLM context analysis to console."""
    print("\n🔍 Context Analysis (AI):")
    print("=" * 60)
    print(f"📋 요약: {analysis['summary']}")
    print(f"📍 상황: {analysis['situation']}")
    print(f"📂 의도: {analysis['intent']}")
    print(f"🎭 감정: {analysis['sentiment']}")
    print(f"✅ 권장 조치:")
    for action in analysis['next_actions']:
        print(f"   - {action}")
    print("=" * 60)

def main():
    p = argparse.ArgumentParser(description="VibeVoice ASR 마이크 테스트")
    p.add_argument("--model_path", default="microsoft/VibeVoice-ASR")
    p.add_argument("--duration", type=int, default=5, help="녹음 시간(초)")
    p.add_argument("--device", default="auto", help="cuda/cpu/auto")
    p.add_argument("--mic", type=int, help="마이크 인덱스")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--keep-audio", action="store_true")
    p.add_argument("--no-analysis", action="store_true", help="Skip LLM context analysis")
    args = p.parse_args()
    
    if args.list_devices:
        list_devices(); sys.exit(0)
    
    device = "cuda" if args.device == "auto" and HAS_CUDA else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not HAS_CUDA:
        print("CUDA 불가, CPU 사용"); device = "cpu"
    print(f"장치: {device}" + (f" ({GPU_NAME})" if device == "cuda" else ""))
    
    print(f"\n모델 로딩: {args.model_path}")
    try:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        asr = VibeVoiceASRInference(model_path=args.model_path, device=device, dtype=dtype, attn_implementation="eager")
    except Exception as e:
        print(f"모델 로딩 실패: {e}"); sys.exit(1)
    
    audio = record(args.duration, device=args.mic)
    if audio is None:
        print("녹음 실패"); sys.exit(1)
    
    tmp = f"temp_mic_{int(time_mod.time())}.wav"
    sf.write(tmp, audio, 16000)
    
    print("\n인식 중...")
    start = time_mod.time()
    try:
        result = asr.transcribe(audio_path=tmp, max_new_tokens=512, do_sample=False)
    except Exception as e:
        print(f"전사 실패: {e}")
        if os.path.exists(tmp): os.remove(tmp)
        sys.exit(1)
    
    print(f"\n{'='*50}\n결과:\n{'='*50}")
    print(result.get("raw_text", "(인식 없음)"))
    print(f"{'='*50}\n처리: {time_mod.time()-start:.2f}초")

    # Display Audio Segments
    segments = result.get("segments", [])
    print_segments(segments)

    # Display Context Analysis (if not skipped)
    if not args.no_analysis and segments:
        try:
            llm_client = LLMClient()
            analysis = llm_client.analyze_call(segments)
            print_analysis(analysis)
        except Exception as e:
            print(f"\n⚠️  Context analysis skipped (error: {e})")
    elif args.no_analysis:
        print("\n⏭️  Context Analysis skipped (--no-analysis flag)")

    if not args.keep_audio and os.path.exists(tmp): os.remove(tmp)

if __name__ == "__main__":
    main()
