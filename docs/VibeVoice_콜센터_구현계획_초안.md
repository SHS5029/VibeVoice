# Microsoft VibeVoice 기반 AI 콜센터 구현 계획 초안

> **작성일**: 2026-01-27
> **기반 기술**: [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) (ASR + TTS 패밀리)
> **라이선스**: MIT — 전체 코드 및 모델 가중치 상용화 가능
> **GitHub Stars**: 22.3k | **월간 다운로드**: 46,933회 (ASR 기준)

---

## 1. 기술 개요 (Technology Overview)

### 1.1 VibeVoice란?

VibeVoice는 Microsoft가 개발한 **오픈소스 음성 AI 모델 패밀리**로, 콜센터에 필요한 핵심 기능을 모듈별로 제공합니다.

| 모델 | 파라미터 | 역할 | 핵심 기능 |
|------|---------|------|----------|
| **VibeVoice-ASR** | 9B | 음성 인식 (STT) | 60분 단일 패스, 화자분리, 타임스탬프, 핫워드 |
| **VibeVoice-Realtime** | 0.5B | 실시간 음성 합성 (TTS) | ~300ms 지연, 스트리밍 입력, 한국어 실험적 지원 |
| **VibeVoice-TTS** | 1.5B | 장문 음성 합성 (TTS) | 90분 생성, 4명 다화자, 고품질 |

### 1.2 콜센터 적용 아키텍처: 캐스케이드 파이프라인

VibeVoice는 PersonaPlex 같은 End-to-End 모델이 아닌 **ASR + LLM + TTS 캐스케이드 방식**입니다.

```
┌──────────────────────────────────────────────────────────────┐
│                   콜센터 AI 파이프라인                         │
│                                                              │
│  고객 음성 ──→ [VibeVoice-ASR] ──→ 텍스트 (Who/When/What)    │
│                    (9B)              │                        │
│                                      ▼                       │
│                              [LLM 오케스트레이터]              │
│                              (GPT-4o / Claude /              │
│                               Llama / Qwen 등)               │
│                                      │                       │
│                                      ▼                       │
│  고객 ←────── [VibeVoice-Realtime] ←── 응답 텍스트            │
│                    (0.5B)                                     │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 PersonaPlex(End-to-End) vs VibeVoice(캐스케이드) 비교

| 항목 | PersonaPlex (E2E) | VibeVoice (캐스케이드) |
|------|-------------------|----------------------|
| **아키텍처** | 단일 모델 (7B) | ASR(9B) + LLM + TTS(0.5B) 조합 |
| **통화 방식** | 전이중 (Full-Duplex) | 반이중 (Half-Duplex) — 추가 구현 필요 |
| **응답 지연** | ~170ms | ~300ms(TTS) + ASR + LLM 처리 = 500ms~1.5s |
| **한국어** | 미지원 | ASR: 50개+ 언어 / TTS: 한국어 실험적 지원 |
| **화자 분리** | 미지원 | 내장 (Who/When/What) |
| **커스터마이징** | 프롬프트 기반 | ASR 핫워드 + LLM 프롬프트 + TTS 음성 선택 |
| **외부 시스템 연동** | 어려움 (E2E) | 용이 (LLM 레이어에서 자유롭게 연동) |
| **통화 분석** | 별도 구현 | ASR 출력이 곧 구조화된 분석 데이터 |
| **LoRA 파인튜닝** | 미확인 | 공식 지원 |
| **라이선스** | MIT + NVIDIA 라이선스 | MIT (전체) |

**VibeVoice의 결정적 장점:**
1. **다국어 지원** — 50개+ 언어 ASR, 한국어 TTS 실험적 지원
2. **비즈니스 로직 유연성** — LLM 레이어에서 CRM/DB/RAG 자유롭게 연동
3. **화자 분리 내장** — 상담원/고객 자동 구분, 통화 분석에 즉시 활용
4. **핫워드 커스터마이징** — 도메인 용어 인식률 대폭 향상
5. **순수 MIT 라이선스** — 상용화 제약 최소

**VibeVoice의 한계:**
1. **반이중 통화** — 전이중 대화(끼어들기, 맞장구) 구현에 추가 엔지니어링 필요
2. **파이프라인 지연** — E2E 대비 응답 지연 누적 (목표: 1초 이내)
3. **TTS 한국어 품질** — 아직 실험적 단계, 파인튜닝 필요 가능

---

## 2. 시스템 아키텍처 (System Architecture)

### 2.1 전체 시스템 구성도

```
                         ┌──────────────────────┐
                         │   PSTN / SIP 게이트    │
                         │   (전화망 연결)        │
                         └──────────┬───────────┘
                                    │ SIP/RTP
                                    ▼
                         ┌──────────────────────┐
                         │   미디어 서버          │
                         │  (FreeSWITCH/Ozone)   │
                         │  - 통화 라우팅         │
                         │  - 오디오 코덱 변환     │
                         └──────────┬───────────┘
                                    │ WebSocket (PCM 16kHz/24kHz)
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                    AI 콜센터 코어 플랫폼                       │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              실시간 대화 파이프라인                    │     │
│  │                                                     │     │
│  │  ┌──────────────┐   ┌───────────────┐              │     │
│  │  │VibeVoice-ASR │   │  VAD 모듈     │              │     │
│  │  │   (9B GPU)   │◄──│ (음성 활동    │◄── 고객 음성  │     │
│  │  │              │   │  감지)        │              │     │
│  │  │ - STT        │   └───────────────┘              │     │
│  │  │ - 화자 분리   │                                  │     │
│  │  │ - 타임스탬프  │                                  │     │
│  │  │ - 핫워드 인식 │                                  │     │
│  │  └──────┬───────┘                                  │     │
│  │         │ 구조화된 텍스트 (Who/When/What)            │     │
│  │         ▼                                          │     │
│  │  ┌──────────────────────────────────────────┐      │     │
│  │  │        LLM 오케스트레이터                  │      │     │
│  │  │                                          │      │     │
│  │  │  ┌────────────┐  ┌────────────────────┐  │      │     │
│  │  │  │ 의도 분류   │  │ 응답 생성           │  │      │     │
│  │  │  │ (Intent)   │  │ (Response Gen.)    │  │      │     │
│  │  │  └────────────┘  └────────────────────┘  │      │     │
│  │  │  ┌────────────┐  ┌────────────────────┐  │      │     │
│  │  │  │ RAG 검색   │  │ Function Calling   │  │      │     │
│  │  │  │ (지식베이스)│  │ (CRM/주문/결제)    │  │      │     │
│  │  │  └────────────┘  └────────────────────┘  │      │     │
│  │  └──────────┬───────────────────────────────┘      │     │
│  │             │ 응답 텍스트 (스트리밍)                  │     │
│  │             ▼                                      │     │
│  │  ┌────────────────────┐                            │     │
│  │  │VibeVoice-Realtime  │──── 에이전트 음성 ──→ 고객  │     │
│  │  │    (0.5B GPU)      │                            │     │
│  │  │ - 스트리밍 TTS     │                            │     │
│  │  │ - ~300ms 지연      │                            │     │
│  │  │ - 한국어 지원       │                            │     │
│  │  └────────────────────┘                            │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              지원 서비스 레이어                       │     │
│  │                                                     │     │
│  │  ┌───────────┐ ┌──────────┐ ┌───────────────────┐  │     │
│  │  │  세션      │ │ 에스컬   │ │  통화 분석         │  │     │
│  │  │  관리자    │ │ 레이션   │ │  엔진             │  │     │
│  │  │           │ │ 관리자   │ │                   │  │     │
│  │  │ - 상태관리│ │ - 인간   │ │ - 감정 분석       │  │     │
│  │  │ - 타임아웃│ │   전환   │ │ - 품질 스코어링   │  │     │
│  │  │ - 이력관리│ │ - 콜백   │ │ - 키워드 추출     │  │     │
│  │  └───────────┘ └──────────┘ └───────────────────┘  │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              데이터 레이어                            │     │
│  │                                                     │     │
│  │  ┌────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐  │     │
│  │  │ CRM    │ │ 지식베이스│ │ 통화   │ │ 분석      │  │     │
│  │  │ 연동   │ │ (Vector  │ │ 로그   │ │ 데이터    │  │     │
│  │  │ API    │ │  DB+RAG) │ │ 저장소 │ │ 웨어하우스│  │     │
│  │  └────────┘ └──────────┘ └────────┘ └───────────┘  │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │          모니터링 & 관리 대시보드                     │     │
│  │  - 실시간 통화 현황    - GPU/인프라 메트릭            │     │
│  │  - 에이전트 성능       - 에스컬레이션 알림            │     │
│  │  - 고객 만족도 추이    - 비용 추적                   │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 실시간 대화 흐름 (Sequence)

```
고객           미디어서버        VAD         ASR(9B)      LLM        TTS(0.5B)
 │               │              │            │            │            │
 │──음성전화──→  │              │            │            │            │
 │               │──오디오──→   │            │            │            │
 │               │              │──음성구간─→│            │            │
 │               │              │            │──텍스트──→ │            │
 │               │              │            │(Who/What)  │            │
 │               │              │            │            │──응답──→   │
 │               │              │            │            │(스트리밍)   │
 │  ◀──────────────────────────────────────────────────── │──음성──→  │
 │  에이전트 음성                                         │            │
 │               │              │            │            │            │
 │  (총 지연: ASR ~200ms + LLM ~300ms + TTS ~300ms ≈ 800ms~1.5s)     │
```

### 2.3 핵심 컴포넌트 상세

#### A. VibeVoice-ASR 서빙 레이어

| 항목 | 상세 |
|------|------|
| **모델** | VibeVoice-ASR 9B (BF16, Safetensors) |
| **기반 프레임워크** | Transformers + vLLM (고속 추론) |
| **입력** | 오디오 스트림 (24kHz) |
| **출력** | 구조화된 전사: `{speaker, timestamp, text}` |
| **특수 기능** | 커스텀 핫워드, 50개+ 언어, LoRA 파인튜닝 |
| **GPU 요구** | NVIDIA A100 80GB (권장) |
| **서빙 옵션** | vLLM 플러그인, Gradio 데모, 직접 추론 |

#### B. LLM 오케스트레이터 (두뇌)

| 옵션 | 장점 | 단점 | 비용 |
|------|------|------|------|
| **GPT-4o (API)** | 최고 품질, Function Calling | 종량제 비용, 외부 의존 | $$$ |
| **Claude 3.5 (API)** | 긴 컨텍스트, 안전성 | 종량제 비용, 외부 의존 | $$$ |
| **Llama 3.1 70B (자체)** | 자체 운영, 비용 예측 | GPU 추가 필요 | $$ (인프라) |
| **Qwen2.5 72B (자체)** | 한중영 우수, 오픈소스 | GPU 추가 필요 | $$ (인프라) |
| **Qwen2.5 7B (자체)** | 경량, 빠른 응답 | 복잡한 추론 제한 | $ (인프라) |

#### C. VibeVoice-Realtime TTS 레이어

| 항목 | 상세 |
|------|------|
| **모델** | VibeVoice-Realtime-0.5B |
| **기반** | Qwen2.5-0.5B + σ-VAE + Diffusion Head |
| **지연시간** | ~300ms (첫 오디오 청크) |
| **스트리밍** | 텍스트 스트리밍 입력 지원 → LLM 토큰 스트리밍과 직결 |
| **한국어** | 실험적 지원 (DE, FR, IT, JP, **KR**, NL, PL, PT, ES) |
| **음성** | 다양한 영어 스타일 11종 + 다국어 9종 |
| **GPU 요구** | 비교적 경량 (0.5B) — 소형 GPU 또는 ASR과 공유 가능 |
| **안전장치** | AI 생성 자동 고지, 워터마크 내장 |

---

## 3. 인프라 요구사항 (Infrastructure Requirements)

### 3.1 하드웨어 사양

#### 최소 구성 (PoC / 소규모)

| 컴포넌트 | GPU | VRAM | CPU | RAM | 역할 |
|---------|-----|------|-----|-----|------|
| ASR 서버 | A100 80GB × 1 | 80GB | 16코어 | 64GB | VibeVoice-ASR (9B) |
| LLM 서버 | — (API 사용) | — | 4코어 | 16GB | GPT-4o / Claude API |
| TTS 서버 | RTX 4090 × 1 | 24GB | 8코어 | 32GB | VibeVoice-Realtime (0.5B) |

> TTS(0.5B)는 경량 모델이므로 소비자급 GPU에서도 구동 가능

#### 권장 구성 (프로덕션 / 중규모)

| 컴포넌트 | GPU | 수량 | 동시 처리 | 역할 |
|---------|-----|------|----------|------|
| ASR 클러스터 | A100 80GB | 4대 | ~8~16 세션 | 실시간 STT |
| LLM 서버 | A100 80GB | 2대 | ~20+ 세션 | Qwen2.5-72B 자체 호스팅 |
| TTS 클러스터 | L40S 48GB | 4대 | ~16~32 세션 | 실시간 TTS |
| API 서버 | — | 2대 | — | 오케스트레이터, 세션 관리 |
| DB 서버 | — | 2대 | — | PostgreSQL, Vector DB |

### 3.2 클라우드 비용 추정

| 구성 | 클라우드 인스턴스 | 월 비용 (USD) | 동시 통화 |
|------|-----------------|-------------|----------|
| **최소 (PoC)** | A100×1 + RTX4090×1 + API | ~$3,500 + API비용 | 1~2 |
| **소규모** | A100×2 + L40S×2 + LLM API | ~$10,000 | 4~8 |
| **중규모** | A100×4 + A100×2(LLM) + L40S×4 | ~$30,000 | 16~32 |
| **대규모** | H100 클러스터 | ~$60,000+ | 50~100+ |

> Reserved Instance 활용 시 30~60% 절감 가능

### 3.3 소프트웨어 스택

```
운영체제       : Ubuntu 22.04+ (Linux 필수)
GPU 드라이버   : NVIDIA Driver 535+
CUDA           : 12.0+
컨테이너       : Docker + NVIDIA Container Toolkit
                 NVIDIA PyTorch Container (nvcr.io/nvidia/pytorch:25.12-py3)
Python         : 3.10+
ML 프레임워크  : PyTorch 2.0+, Transformers, Safetensors
추론 최적화    : vLLM (ASR 추론 가속)
오디오 처리    : ffmpeg, libopus
전화 연동      : FreeSWITCH / Twilio / Amazon Connect
API 서버       : Python FastAPI / Go
오케스트레이션 : Kubernetes + NVIDIA GPU Operator
모니터링       : Prometheus + Grafana
로깅           : ELK Stack
CI/CD          : GitHub Actions + ArgoCD
```

---

## 4. 구현 단계별 계획 (Implementation Phases)

### Phase 1: PoC — 기술 검증

**목표**: VibeVoice-ASR + LLM + VibeVoice-Realtime 파이프라인 기본 동작 검증

#### 1-1. 환경 구축

```bash
# Docker 환경 (권장)
sudo docker run --gpus all -it \
  nvcr.io/nvidia/pytorch:25.12-py3 bash

# VibeVoice 설치
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
apt install ffmpeg -y
```

#### 1-2. ASR 테스트 — 콜센터 녹음 전사

```bash
# 파일 기반 추론 테스트
python demo/vibevoice_asr_inference_from_file.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio_files sample_call_recording.wav
```

**예상 출력 (구조화된 전사):**
```json
{
  "transcription": [
    {
      "speaker": "Speaker_1",
      "start_time": 0.0,
      "end_time": 3.2,
      "text": "안녕하세요, 고객센터입니다. 무엇을 도와드릴까요?"
    },
    {
      "speaker": "Speaker_2",
      "start_time": 3.5,
      "end_time": 8.1,
      "text": "네, 지난주에 주문한 제품이 아직 도착하지 않아서 문의드립니다."
    }
  ]
}
```

#### 1-3. 핫워드 설정 테스트

```python
# 도메인 핫워드 예시 — 금융 콜센터
hotwords = [
    "적금",        # 자주 인식 오류 나는 용어
    "CMA 통장",
    "ISA 계좌",
    "퇴직연금 IRP",
    "비대면 계좌개설",
    "본인확인서비스",
]

# ASR 추론 시 핫워드 전달
result = model.transcribe(
    audio=customer_audio,
    hotwords=hotwords,
    language="ko"
)
```

#### 1-4. TTS 테스트 — 에이전트 음성 생성

```python
# VibeVoice-Realtime 실시간 TTS
from vibevoice import VibeVoiceRealtime

tts = VibeVoiceRealtime(model_path="microsoft/VibeVoice-Realtime-0.5B")

# 한국어 음성 생성 (실험적)
audio_stream = tts.synthesize_streaming(
    text="네, 고객님. 주문번호 확인해드리겠습니다. 잠시만 기다려주세요.",
    language="ko",
    speaker="ko_female_01"
)
```

#### 1-5. 전체 파이프라인 PoC

```python
import asyncio
from vibevoice_asr import VibeVoiceASR
from vibevoice_realtime import VibeVoiceRealtime
from llm_client import LLMClient  # GPT-4o 또는 자체 LLM

class CallCenterPipeline:
    """VibeVoice 기반 콜센터 AI 파이프라인 PoC"""

    def __init__(self):
        self.asr = VibeVoiceASR(model_path="microsoft/VibeVoice-ASR")
        self.tts = VibeVoiceRealtime(model_path="microsoft/VibeVoice-Realtime-0.5B")
        self.llm = LLMClient(model="gpt-4o")  # 또는 자체 LLM

        # 콜센터 핫워드
        self.hotwords = ["적금", "해지", "이체", "대출", "카드"]

    async def process_call(self, audio_stream):
        """단일 통화 처리 파이프라인"""

        # 1단계: ASR — 고객 음성 → 텍스트
        transcription = await self.asr.transcribe_stream(
            audio=audio_stream,
            hotwords=self.hotwords,
            language="ko"
        )

        customer_text = transcription["text"]
        speaker = transcription["speaker"]

        # 2단계: LLM — 텍스트 → 응답 생성
        response_stream = self.llm.generate_stream(
            system_prompt=self.build_system_prompt(),
            user_message=customer_text,
            tools=self.get_available_tools()  # CRM 조회, 주문 확인 등
        )

        # 3단계: TTS — 응답 텍스트 → 에이전트 음성 (스트리밍)
        async for text_chunk in response_stream:
            audio_chunk = await self.tts.synthesize_chunk(
                text=text_chunk,
                language="ko"
            )
            yield audio_chunk  # 고객에게 실시간 전송

    def build_system_prompt(self):
        return """
        당신은 [회사명] 고객센터 AI 상담원입니다.
        이름: [에이전트명]
        성격: 친절하고 전문적이며 공감 능력이 뛰어남

        업무 범위:
        - 주문 조회 및 배송 상태 확인
        - 제품 문의 및 기술 지원
        - 반품/교환 접수
        - 기본적인 결제 문의

        규칙:
        1. 항상 정중한 존댓말 사용
        2. 고객의 감정에 공감하며 응대
        3. 정확한 정보만 전달 (모르는 경우 솔직히 안내)
        4. 복잡한 문의는 인간 상담원 연결 안내
        5. 통화 마무리 시 처리 내용 요약
        """
```

**Phase 1 산출물:**
- ASR 한국어 인식률 측정 (WER/CER)
- TTS 한국어 음성 품질 평가 (MOS)
- 전체 파이프라인 응답 지연시간 측정
- Go/No-Go 의사결정 보고서

---

### Phase 2: 실시간 전화 연동

**목표**: 실제 전화 통화를 VibeVoice 파이프라인에 연결

#### 2-1. 전화 연동 아키텍처

```python
import asyncio
import websockets
from call_bridge import SIPHandler

class RealtimeCallHandler:
    """전화 통화 ↔ VibeVoice 파이프라인 연결"""

    def __init__(self, pipeline: CallCenterPipeline):
        self.pipeline = pipeline
        self.vad = VoiceActivityDetector(
            threshold=0.5,
            min_speech_duration=0.3,  # 최소 발화 길이
            silence_duration=0.8     # 발화 종료 판단 (묵음 길이)
        )

    async def handle_call(self, sip_session):
        """수신 전화 처리"""

        # 인사 메시지 재생
        greeting = await self.pipeline.tts.synthesize(
            "안녕하세요, [회사명] 고객센터입니다. 무엇을 도와드릴까요?",
            language="ko"
        )
        await sip_session.play_audio(greeting)

        # 대화 루프
        while sip_session.is_active:
            # VAD로 고객 발화 구간 감지
            speech_segment = await self.vad.detect_speech(
                sip_session.audio_stream
            )

            if speech_segment is None:
                continue

            # ASR → LLM → TTS 파이프라인 실행
            async for audio_chunk in self.pipeline.process_call(speech_segment):
                await sip_session.send_audio(audio_chunk)

            # 에스컬레이션 확인
            if self.pipeline.should_escalate():
                await self.transfer_to_human(sip_session)
                break

    async def transfer_to_human(self, sip_session):
        """인간 상담원 전환"""
        transfer_msg = await self.pipeline.tts.synthesize(
            "더 정확한 도움을 위해 전문 상담원에게 연결해드리겠습니다. "
            "잠시만 기다려주세요.",
            language="ko"
        )
        await sip_session.play_audio(transfer_msg)
        await sip_session.transfer_to_queue("human_agents")
```

#### 2-2. VAD (Voice Activity Detection) 전략

콜센터에서 자연스러운 대화를 위한 VAD 설정이 핵심입니다.

```python
class AdaptiveVAD:
    """콜센터 최적화 음성 활동 감지"""

    def __init__(self):
        self.config = {
            "min_speech_ms": 300,      # 최소 발화 길이 (짧은 대답도 포착)
            "max_silence_ms": 800,     # 발화 종료 판단 묵음 길이
            "prefix_padding_ms": 200,  # 발화 시작 전 여유분 (잘림 방지)
            "energy_threshold": 0.4,   # 에너지 임계값
        }

    async def detect_speech_end(self, audio_stream):
        """
        고객 발화가 끝났는지 판단
        - 너무 빨리 끊으면: 고객 말이 잘림
        - 너무 늦게 끊으면: 응답 지연 체감
        → 800ms 묵음이 최적 (콜센터 실증 결과 기반)
        """
        silence_duration = 0
        speech_buffer = []

        async for chunk in audio_stream:
            if self.is_speech(chunk):
                silence_duration = 0
                speech_buffer.append(chunk)
            else:
                silence_duration += len(chunk) / self.sample_rate
                if silence_duration >= self.config["max_silence_ms"] / 1000:
                    return speech_buffer

        return speech_buffer
```

#### 2-3. 전화 시스템 연동 옵션

| 옵션 | 장점 | 단점 | 적합 규모 |
|------|------|------|----------|
| **Twilio Media Streams** | WebSocket 기반, 빠른 구축 | 종량제 비용 | 소~중규모 |
| **Amazon Connect** | AWS 통합, Lex/Lambda 연동 | AWS 종속 | 중규모 |
| **FreeSWITCH + mod_ws** | 무료, 완전 커스터마이징 | 운영 복잡 | 대규모 |
| **Asterisk + ARI** | 무료, 풍부한 생태계 | 레거시 느낌 | 소~중규모 |
| **한국 VoIP: KT/SKT API** | 국내 번호, 규제 준수 | 제한적 API | 국내 전용 |

**Phase 2 산출물:**
- 실시간 전화 ↔ AI 파이프라인 프로토타입
- 음성 품질 측정 (PESQ, POLQA)
- End-to-End 지연시간: 목표 1초 이내
- 동시 통화 부하 테스트 결과

---

### Phase 3: 비즈니스 로직 통합

**목표**: CRM, 지식베이스, 업무 시스템과 완전 연동

#### 3-1. LLM Function Calling 기반 업무 처리

VibeVoice 캐스케이드 아키텍처의 **최대 장점**: LLM 레이어에서 자유롭게 외부 시스템을 호출할 수 있습니다.

```python
# LLM에 제공할 도구(Tool) 정의
CALL_CENTER_TOOLS = [
    {
        "name": "lookup_order",
        "description": "주문번호 또는 고객명으로 주문 상태를 조회합니다",
        "parameters": {
            "order_id": {"type": "string", "description": "주문번호"},
            "customer_name": {"type": "string", "description": "고객 이름"}
        }
    },
    {
        "name": "check_delivery",
        "description": "배송 현재 위치와 예상 도착일을 확인합니다",
        "parameters": {
            "tracking_number": {"type": "string"}
        }
    },
    {
        "name": "process_return",
        "description": "반품/교환을 접수합니다",
        "parameters": {
            "order_id": {"type": "string"},
            "reason": {"type": "string"},
            "return_type": {"type": "string", "enum": ["반품", "교환"]}
        }
    },
    {
        "name": "search_faq",
        "description": "자주 묻는 질문에서 관련 답변을 검색합니다",
        "parameters": {
            "query": {"type": "string"}
        }
    },
    {
        "name": "update_customer_info",
        "description": "고객 연락처, 주소 등 정보를 업데이트합니다",
        "parameters": {
            "customer_id": {"type": "string"},
            "field": {"type": "string"},
            "new_value": {"type": "string"}
        }
    },
    {
        "name": "escalate_to_human",
        "description": "인간 상담원에게 통화를 전환합니다",
        "parameters": {
            "reason": {"type": "string"},
            "priority": {"type": "string", "enum": ["일반", "긴급"]}
        }
    }
]
```

#### 3-2. RAG 기반 지식베이스

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class CallCenterKnowledgeBase:
    """콜센터 지식베이스 — 제품 정보, FAQ, 정책 문서"""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3"  # 다국어 임베딩
        )
        self.vectorstore = Chroma(
            collection_name="call_center_kb",
            embedding_function=self.embeddings,
            persist_directory="./kb_data"
        )

    def index_documents(self, documents: list):
        """문서 색인 — 제품 매뉴얼, FAQ, 정책 등"""
        self.vectorstore.add_documents(documents)

    def search(self, query: str, k: int = 3) -> list:
        """관련 문서 검색"""
        return self.vectorstore.similarity_search(query, k=k)

    # 색인할 문서 카테고리:
    # - 제품 카탈로그 및 사양
    # - 가격/할인 정책
    # - 반품/교환/환불 정책
    # - 배송 정책 및 FAQ
    # - 기술 지원 매뉴얼
    # - 서비스 이용약관
```

#### 3-3. 통화 분석 엔진

VibeVoice-ASR의 **Who/When/What 구조화 출력**을 활용한 자동 분석:

```python
class CallAnalytics:
    """통화 완료 후 자동 분석 — ASR 출력 직접 활용"""

    def analyze_call(self, transcription: dict) -> dict:
        """
        VibeVoice-ASR의 구조화된 출력을 직접 분석에 활용
        - Who: 화자 분리 → 상담원 vs 고객 발화 비율
        - When: 타임스탬프 → 응답 시간, 묵음 구간 분석
        - What: 전사 텍스트 → 감정 분석, 키워드 추출
        """
        return {
            "call_duration": self.calc_duration(transcription),
            "speaker_ratio": self.calc_speaker_ratio(transcription),
            "response_times": self.calc_response_times(transcription),
            "sentiment_flow": self.analyze_sentiment_flow(transcription),
            "key_topics": self.extract_topics(transcription),
            "resolution_status": self.detect_resolution(transcription),
            "compliance_check": self.check_compliance(transcription),
            "quality_score": self.calculate_quality_score(transcription),
        }

    def calc_speaker_ratio(self, transcription):
        """화자별 발화 비율 — 상담원이 너무 많이 말하진 않는지"""
        agent_time = sum(
            seg["end_time"] - seg["start_time"]
            for seg in transcription
            if seg["speaker"] == "agent"
        )
        customer_time = sum(
            seg["end_time"] - seg["start_time"]
            for seg in transcription
            if seg["speaker"] == "customer"
        )
        return {"agent": agent_time, "customer": customer_time}

    def calc_response_times(self, transcription):
        """고객 발화 종료 → 상담원 응답 시작 간격 측정"""
        response_times = []
        for i in range(len(transcription) - 1):
            if (transcription[i]["speaker"] == "customer" and
                transcription[i+1]["speaker"] == "agent"):
                gap = transcription[i+1]["start_time"] - transcription[i]["end_time"]
                response_times.append(gap)
        return {
            "avg": sum(response_times) / len(response_times) if response_times else 0,
            "max": max(response_times) if response_times else 0,
            "details": response_times
        }
```

#### 3-4. 에스컬레이션 시스템

```python
class EscalationManager:
    """AI → 인간 상담원 전환 관리"""

    RULES = {
        "explicit_request": {
            "description": "고객이 직접 인간 상담원을 요청",
            "keywords": ["사람", "상담원", "직원", "매니저", "책임자"],
            "priority": "즉시"
        },
        "repeated_failure": {
            "description": "동일 문의 3회 이상 반복 (해결 불가)",
            "threshold": 3,
            "priority": "높음"
        },
        "negative_sentiment": {
            "description": "고객 감정 점수가 임계값 이하",
            "threshold": -0.7,
            "priority": "높음"
        },
        "sensitive_topics": {
            "description": "법적/금전적 민감 주제",
            "keywords": ["소송", "변호사", "환불", "보상", "피해"],
            "priority": "높음"
        },
        "timeout": {
            "description": "통화 시간 15분 초과",
            "threshold_minutes": 15,
            "priority": "보통"
        }
    }

    async def evaluate(self, session) -> dict:
        """에스컬레이션 필요 여부 실시간 평가"""
        for rule_name, rule in self.RULES.items():
            if self.check_rule(session, rule_name, rule):
                return {
                    "should_escalate": True,
                    "reason": rule["description"],
                    "priority": rule["priority"],
                    "call_summary": self.generate_summary(session)
                }
        return {"should_escalate": False}

    def generate_summary(self, session) -> str:
        """인간 상담원에게 전달할 통화 요약 자동 생성"""
        # LLM을 활용한 대화 요약
        return self.llm.summarize(
            session.transcript,
            prompt="이 통화 내용을 인간 상담원이 인수받을 수 있도록 "
                   "핵심 문의사항, 고객 감정, 시도된 해결책을 요약해줘."
        )
```

**Phase 3 산출물:**
- CRM/ERP 연동 API 및 LLM Function Calling 완성
- RAG 기반 지식베이스 색인 및 검색 파이프라인
- 에스컬레이션 워크플로우 완성
- 통화 자동 분석 대시보드 프로토타입

---

### Phase 4: 한국어 최적화

**목표**: 한국어 콜센터 환경에 특화된 품질 향상

#### 4-1. ASR 한국어 파인튜닝 (LoRA)

VibeVoice-ASR은 공식적으로 LoRA 파인튜닝을 지원합니다.

```bash
# 파인튜닝 코드 (VibeVoice 공식 제공)
cd VibeVoice/finetuning-asr/

# 한국어 콜센터 데이터로 LoRA 파인튜닝
python train.py \
  --base_model microsoft/VibeVoice-ASR \
  --train_data ./korean_callcenter_data/ \
  --language ko \
  --lora_rank 16 \
  --lora_alpha 32 \
  --epochs 5 \
  --batch_size 4
```

**필요한 한국어 학습 데이터:**

| 데이터 소스 | 설명 | 규모 |
|-----------|------|------|
| **AIHub 한국어 대화 데이터** | 정부 제공 무료 데이터셋 | 1,000시간+ |
| **자사 콜센터 녹음** | 실제 업무 대화 (동의 필요) | 100~500시간 |
| **도메인 용어 사전** | 핫워드용 전문 용어 | 500~2,000개 |
| **합성 데이터** | TTS로 생성한 시나리오 대화 | 보충용 |

#### 4-2. TTS 한국어 음성 품질 개선

```python
# 현재 상태: 실험적 한국어 지원
# 전략 1: VibeVoice-Realtime 한국어 파인튜닝 (데이터 확보 시)
# 전략 2: 한국어 전문 TTS와 하이브리드 — 품질 우선 시

# 대안 TTS 옵션 (한국어 품질이 불충분할 경우)
KOREAN_TTS_ALTERNATIVES = {
    "azure_tts": {
        "provider": "Microsoft Azure Cognitive Services",
        "quality": "높음",
        "cost": "종량제",
        "latency": "~200ms",
        "note": "한국어 네이티브 품질"
    },
    "google_cloud_tts": {
        "provider": "Google Cloud",
        "quality": "높음",
        "cost": "종량제",
        "latency": "~200ms",
        "note": "한국어 WaveNet 음성"
    },
    "coqui_tts": {
        "provider": "오픈소스 (Coqui)",
        "quality": "중간",
        "cost": "무료",
        "latency": "~500ms",
        "note": "자체 호스팅 가능"
    }
}
```

#### 4-3. 콜센터 핫워드 사전 구축

```python
# 업종별 핫워드 사전 예시

HOTWORDS_BANKING = [
    # 상품명
    "적금", "예금", "CMA", "ISA", "IRP", "퇴직연금",
    "주택담보대출", "신용대출", "마이너스통장",
    # 서비스
    "인터넷뱅킹", "모바일뱅킹", "비대면", "공인인증서",
    "OTP", "보안카드", "이체한도",
    # 고유명사
    "토스뱅크", "카카오뱅크", "케이뱅크",
]

HOTWORDS_ECOMMERCE = [
    "무료배송", "당일배송", "새벽배송", "반품", "교환",
    "포인트", "쿠폰", "할인코드", "회원등급",
    "로켓배송", "네이버페이", "카카오페이",
]

HOTWORDS_TELECOM = [
    "5G", "LTE", "데이터 무제한", "요금제", "번호이동",
    "기기변경", "약정", "위약금", "로밍",
    "eSIM", "유심", "알뜰폰",
]
```

**Phase 4 산출물:**
- LoRA 파인튜닝된 한국어 ASR 모델
- 한국어 WER/CER 벤치마크 (목표: WER < 10%)
- 업종별 핫워드 사전
- TTS 한국어 품질 평가 및 대안 선정

---

### Phase 5: 프로덕션 배포 및 운영

**목표**: 안정적인 프로덕션 환경 구축 및 스케일링

#### 5-1. 컨테이너 구성

```yaml
# docker-compose.yaml
version: '3.8'

services:
  # VibeVoice-ASR 서버
  vibevoice-asr:
    image: nvcr.io/nvidia/pytorch:25.12-py3
    command: >
      python -m vibevoice.asr.server
      --model_path microsoft/VibeVoice-ASR
      --port 8001
      --language ko
      --hotwords /app/hotwords/banking.json
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8001:8001"
    volumes:
      - ./hotwords:/app/hotwords
      - ./models:/app/models

  # VibeVoice-Realtime TTS 서버
  vibevoice-tts:
    image: nvcr.io/nvidia/pytorch:25.12-py3
    command: >
      python -m vibevoice.realtime.server
      --model_path microsoft/VibeVoice-Realtime-0.5B
      --port 8002
      --language ko
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8002:8002"

  # LLM 오케스트레이터
  orchestrator:
    build: ./orchestrator
    ports:
      - "8000:8000"
    environment:
      - ASR_URL=http://vibevoice-asr:8001
      - TTS_URL=http://vibevoice-tts:8002
      - LLM_PROVIDER=openai  # 또는 self-hosted
      - LLM_API_KEY=${LLM_API_KEY}
      - CRM_API_URL=${CRM_API_URL}
      - DB_URL=${DB_URL}
    depends_on:
      - vibevoice-asr
      - vibevoice-tts

  # SIP 미디어 브릿지
  call-bridge:
    build: ./call-bridge
    ports:
      - "5060:5060/udp"
      - "8080:8080"
    depends_on:
      - orchestrator

  # 모니터링
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/dashboards:/var/lib/grafana/dashboards

  # 통화 로그 DB
  postgres:
    image: postgres:16
    environment:
      - POSTGRES_DB=callcenter
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # 벡터 DB (RAG)
  chromadb:
    image: chromadb/chroma
    ports:
      - "8200:8000"
    volumes:
      - chromadata:/chroma/chroma

volumes:
  pgdata:
  chromadata:
```

#### 5-2. Kubernetes 배포 (대규모)

```yaml
# ASR GPU Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vibevoice-asr
  namespace: callcenter
spec:
  replicas: 4
  selector:
    matchLabels:
      app: vibevoice-asr
  template:
    spec:
      containers:
      - name: asr
        image: callcenter/vibevoice-asr:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "80Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "64Gi"
        ports:
        - containerPort: 8001
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 10
      nodeSelector:
        nvidia.com/gpu.product: "NVIDIA-A100-SXM4-80GB"
---
# TTS GPU Deployment (경량 — 더 많은 인스턴스 가능)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vibevoice-tts
  namespace: callcenter
spec:
  replicas: 8
  selector:
    matchLabels:
      app: vibevoice-tts
  template:
    spec:
      containers:
      - name: tts
        image: callcenter/vibevoice-tts:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
        ports:
        - containerPort: 8002
      nodeSelector:
        nvidia.com/gpu.product: "NVIDIA-L40S"
---
# HPA (동시 통화량 기반 오토스케일링)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: asr-hpa
  namespace: callcenter
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vibevoice-asr
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: active_sessions
      target:
        type: AverageValue
        averageValue: "2"
```

#### 5-3. 모니터링 메트릭

| 카테고리 | 메트릭 | 임계값 | 알림 레벨 |
|---------|--------|--------|----------|
| **ASR 성능** | 인식 지연시간 | > 500ms | Warning |
| **ASR 품질** | 한국어 WER | > 15% | Critical |
| **TTS 성능** | 첫 청크 지연 | > 500ms | Warning |
| **LLM 성능** | 응답 생성 지연 | > 1s | Warning |
| **E2E 지연** | 고객발화→응답시작 | > 2s | Critical |
| **GPU** | VRAM 사용률 | > 95% | Warning |
| **GPU** | GPU 사용률 | > 90% | Warning |
| **서비스** | 통화 완료율 | < 70% | Critical |
| **서비스** | 에스컬레이션 비율 | > 30% | Warning |
| **고객** | CSAT 만족도 | < 3.5/5 | Warning |
| **비용** | GPU 시간당 통화 수 | < 기준치 | Info |

---

## 5. 업종별 프롬프트 & 핫워드 설계 가이드

### 5.1 금융/은행 콜센터

```
[시스템 프롬프트]
당신은 [은행명] AI 고객센터 상담원 '민지'입니다.

역할: 금융 상품 문의, 계좌 관련 업무, 카드 분실 신고 처리
성격: 신뢰감 있고, 정확하며, 고객의 금융 걱정에 공감

보안 규칙:
- 생년월일, 계좌번호 뒷 4자리로 본인확인 먼저 수행
- 비밀번호, 보안카드 전체번호는 절대 요청 금지
- 이체/해지 등 중요 거래는 인간 상담원 연결 안내

응대 가능 업무:
1. 계좌 잔액/거래내역 조회
2. 금융 상품 안내 (예금, 적금, 펀드)
3. 카드 분실/도난 신고 접수
4. 인터넷/모바일뱅킹 이용 안내
5. ATM 위치 및 운영시간 안내

핫워드: 적금, 예금, CMA, ISA, IRP, 퇴직연금, 주담대, 신용대출,
       마이너스통장, OTP, 보안카드, 이체한도, 비대면
```

### 5.2 이커머스 고객센터

```
[시스템 프롬프트]
당신은 [쇼핑몰명] AI 고객센터 상담원 '하늘'입니다.

역할: 주문/배송 문의, 반품/교환 처리, 제품 문의 응대
성격: 밝고 친절하며, 문제를 적극적으로 해결하는 자세

반품/교환 정책:
- 수령 후 7일 이내 반품 가능
- 단순 변심: 왕복 배송비 고객 부담
- 제품 불량: 무료 반품 + 교환 또는 환불
- 맞춤 제작 상품: 반품 불가

핫워드: 무료배송, 당일배송, 새벽배송, 반품, 교환, 포인트, 쿠폰,
       할인코드, 회원등급, 네이버페이, 카카오페이
```

### 5.3 통신사 고객센터

```
[시스템 프롬프트]
당신은 [통신사명] AI 고객센터 상담원 '서준'입니다.

역할: 요금제 안내/변경, 기기 관련 문의, 기술 지원
성격: 기술적 내용을 쉽게 설명하며, 인내심 있는 응대

가능 업무:
1. 요금제 조회 및 변경 안내
2. 데이터 사용량 확인
3. 기기변경/번호이동 절차 안내
4. 기본 기술 지원 (재부팅, 설정 초기화, APN 설정)
5. 로밍 서비스 안내

핫워드: 5G, LTE, 데이터무제한, 요금제, 번호이동, 기기변경,
       약정, 위약금, 로밍, eSIM, 유심, 알뜰폰
```

### 5.4 음성 선택 가이드 (VibeVoice-Realtime)

| 음성 ID | 언어 | 스타일 | 적합 업종 |
|---------|------|--------|----------|
| en_female_warm | 영어 (여성) | 따뜻하고 전문적 | 금융, 의료 |
| en_male_professional | 영어 (남성) | 신뢰감 있는 | 법률, 보험 |
| en_female_energetic | 영어 (여성) | 활기찬 | 이커머스, 여행 |
| ko_female_01 | 한국어 (여성) | 실험적 지원 | 범용 (품질 검증 필요) |
| ko_male_01 | 한국어 (남성) | 실험적 지원 | 범용 (품질 검증 필요) |

> 한국어 음성 품질이 불충분할 경우 Azure TTS 또는 Google Cloud TTS를 TTS 레이어 대안으로 사용

---

## 6. 리스크 및 대응 방안

### 6.1 기술적 리스크

| 리스크 | 영향도 | 발생확률 | 대응 방안 |
|--------|--------|---------|----------|
| **파이프라인 지연 누적** | 높음 | 높음 | 스트리밍 최적화, ASR→LLM→TTS 동시 스트리밍, VAD 튜닝 |
| **한국어 ASR 인식률 부족** | 높음 | 중간 | LoRA 파인튜닝 + 핫워드 적극 활용 + AIHub 데이터 활용 |
| **한국어 TTS 품질 부족** | 높음 | 높음 | VibeVoice-Realtime 파인튜닝 또는 Azure/Google TTS 대체 |
| **환각(Hallucination)** | 높음 | 중간 | RAG 기반 사실 검증, LLM 응답 가드레일, 확신도 낮으면 에스컬레이션 |
| **GPU 비용 과다** | 높음 | 높음 | vLLM 최적화, 양자화(INT8/INT4), Reserved Instance |
| **반이중 한계 (끼어들기)** | 중간 | 확실 | 연속 VAD 모니터링 + 즉시 TTS 중단 로직 |
| **서비스 장애** | 높음 | 낮음 | 이중화, 자동 인간 상담원 전환 Fallback |

### 6.2 반이중 한계 극복 전략

캐스케이드 파이프라인의 가장 큰 약점인 **끼어들기(Barge-in) 처리**:

```python
class BargeInHandler:
    """고객 끼어들기 감지 및 처리"""

    async def monitor_during_playback(self, sip_session, tts_stream):
        """
        TTS 재생 중에도 고객 음성을 계속 모니터링
        → 끼어들기 감지 시 즉시 TTS 중단
        """
        async for audio_chunk in tts_stream:
            # TTS 음성 재생
            await sip_session.send_audio(audio_chunk)

            # 동시에 고객 오디오 확인
            customer_audio = await sip_session.peek_audio(timeout=0.05)
            if customer_audio and self.vad.is_speech(customer_audio):
                # 끼어들기 감지!
                tts_stream.cancel()  # TTS 즉시 중단
                return customer_audio  # 고객 발화를 ASR로 전달

        return None  # 정상 완료 (끼어들기 없음)
```

### 6.3 운영적 리스크

| 리스크 | 대응 방안 |
|--------|----------|
| **고객 AI 거부감** | "AI 상담원입니다" 사전 안내, 인간 상담원 연결 옵션 항상 제공 |
| **개인정보 보호** | On-premise 배포, 통화 녹음 암호화, GDPR/PIPA 준수 |
| **법적 규제** | AI 상담 사전 고지 의무 준수, 금융/의료 규제 확인 |
| **품질 편차** | 지속적 LoRA 파인튜닝, A/B 테스트, 주간 품질 리뷰 |
| **모델 업데이트** | Microsoft 업데이트 추적, 자체 파인튜닝 재적용 파이프라인 |

### 6.4 비용 분석 — ROI 추정

```
[비용 비교: 인간 상담원 20명 vs VibeVoice AI 콜센터]

═══════════════════════════════════════════
인간 상담원 (20명 기준, 연간)
═══════════════════════════════════════════
  인건비          : 20명 × ₩3,500만 = ₩7.0억
  교육/복지       : ₩1.4억
  시설/장비       : ₩0.5억
  ───────────────────────────
  합계            : ~₩8.9억/년

═══════════════════════════════════════════
VibeVoice AI 콜센터 (동시 20통화, 연간)
═══════════════════════════════════════════
  ASR GPU (A100×4)       : ~₩1.4억 (Reserved)
  TTS GPU (L40S×4)       : ~₩0.6억 (Reserved)
  LLM (API 또는 자체GPU) : ~₩0.8억
  전화/네트워크           : ~₩0.3억
  개발/운영 인력 (3명)    : ~₩1.5억
  클라우드/기타           : ~₩0.3억
  ───────────────────────────
  합계                    : ~₩4.9억/년

═══════════════════════════════════════════
  예상 절감: ~₩4.0억/년 (45% 절감)
  초기 개발비: ~₩2~3억 (1회성)
  손익분기점: 약 6~9개월 후
  24/7 운영 시 인간 대비 효율 3배+
═══════════════════════════════════════════

※ 추가 이점:
  - 24/7 무중단 운영 (야간/휴일 추가 인건비 없음)
  - 동시 확장 용이 (GPU 추가만으로 용량 증대)
  - 일관된 응대 품질 (감정 기복 없음)
  - 통화 분석 자동화 (별도 QA 인력 불필요)
```

---

## 7. 프로젝트 마일스톤

```
Phase 1: PoC (개념 증명)
├── VibeVoice-ASR + Realtime 환경 구축
├── 한국어 ASR 인식률 테스트 (기본 + 핫워드)
├── 한국어 TTS 품질 평가
├── 전체 파이프라인 (ASR→LLM→TTS) 동작 검증
├── 응답 지연시간 측정 (목표: < 1.5초)
└── Go/No-Go 의사결정

Phase 2: MVP (최소 기능 제품)
├── 전화망(SIP) ↔ 파이프라인 실시간 연결
├── VAD 및 끼어들기(Barge-in) 처리
├── 기본 에스컬레이션 (인간 상담원 전환)
├── 통화 녹음 및 로깅
└── 내부 파일럿 테스트 (10~20통화/일)

Phase 3: 비즈니스 통합
├── LLM Function Calling + CRM/ERP 연동
├── RAG 기반 지식베이스 구축
├── 에스컬레이션 규칙 엔진
├── 통화 자동 분석 (감정, 품질, 토픽)
├── 관리 대시보드 구축
└── 제한적 외부 파일럿

Phase 4: 한국어 최적화
├── LoRA 파인튜닝 (ASR 한국어 특화)
├── 업종별 핫워드 사전 구축
├── TTS 한국어 음성 품질 개선 또는 대안 적용
├── 한국어 시나리오 집중 검증
└── WER < 10% 달성 확인

Phase 5: 프로덕션 & 스케일링
├── Kubernetes + GPU 오토스케일링 배포
├── 모니터링/알림 체계 구축
├── vLLM 추론 최적화
├── 전체 고객 대상 서비스 오픈
├── 지속적 품질 개선 루프 구축
└── 비용 최적화 (양자화, 배치 등)
```

---

## 8. 기술 스택 요약

| 레이어 | 기술 |
|--------|------|
| **ASR** | Microsoft VibeVoice-ASR (9B) — 화자분리, 타임스탬프, 핫워드 |
| **TTS** | Microsoft VibeVoice-Realtime (0.5B) — 실시간 스트리밍 |
| **LLM** | GPT-4o / Claude / Qwen2.5 / Llama 3.1 (선택) |
| **추론 최적화** | vLLM (ASR 가속), Flash Attention |
| **GPU** | NVIDIA A100/H100 (ASR), L40S/RTX4090 (TTS) |
| **전화 연동** | FreeSWITCH / Twilio / Amazon Connect |
| **백엔드** | Python (FastAPI), Go (미디어 브릿지) |
| **지식베이스** | LangChain + ChromaDB / Pinecone (RAG) |
| **CRM 연동** | REST API / gRPC |
| **데이터베이스** | PostgreSQL (통화 로그), ChromaDB (벡터 검색) |
| **모니터링** | Prometheus + Grafana |
| **로깅** | ELK Stack |
| **컨테이너** | Docker + NVIDIA Container Toolkit |
| **오케스트레이션** | Kubernetes + NVIDIA GPU Operator + KEDA |
| **CI/CD** | GitHub Actions + ArgoCD |
| **보안** | TLS/SSL, 데이터 암호화, RBAC |

---

## 9. VibeVoice 모델 패밀리 활용 전략

VibeVoice는 ASR뿐 아니라 TTS 패밀리 전체를 콜센터에 활용할 수 있습니다.

| 모델 | 콜센터 활용 시나리오 |
|------|---------------------|
| **VibeVoice-ASR (9B)** | 실시간 통화 전사, 화자 분리, 통화 분석 |
| **VibeVoice-Realtime (0.5B)** | 실시간 에이전트 음성 응답 (저지연) |
| **VibeVoice-TTS (1.5B)** | 통화 녹음 재현, 트레이닝 자료 생성, 합성 데이터 |

```
[실시간 통화]          [사후 분석]           [트레이닝]

ASR(9B)               ASR(9B)              TTS(1.5B)
  ↓                     ↓                    ↓
LLM                   분석 엔진            시나리오 음성
  ↓                     ↓                    ↓
Realtime(0.5B)        리포트/대시보드       교육 자료 생성
  ↓
고객 응답
```

---

## 10. 참고 자료

- [Microsoft VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [VibeVoice-ASR (Hugging Face)](https://huggingface.co/microsoft/VibeVoice-ASR)
- [VibeVoice-Realtime-0.5B (Hugging Face)](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- [VibeVoice-TTS 1.5B (Hugging Face)](https://huggingface.co/microsoft/VibeVoice-1.5B)
- [VibeVoice 공식 프로젝트 페이지](https://microsoft.github.io/VibeVoice/)
- [VibeVoice-ASR 상세 문서](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md)
- [VibeVoice-ASR 소개 (MarkTechPost)](https://www.marktechpost.com/2026/01/22/microsoft-releases-vibevoice-asr-a-unified-speech-to-text-model-designed-to-handle-60-minute-long-form-audio-in-a-single-pass/)
- [VibeVoice 기술 논문](https://arxiv.org/pdf/2508.19205)
- [VibeVoice-ASR 라이브 데모](https://aka.ms/vibevoice-asr)

---

> **문서 상태**: 초안 (Draft)
> **다음 단계**: 이해관계자 리뷰 → Phase 1 PoC 착수 → 한국어 품질 검증
> **핵심 의사결정 포인트**: 한국어 TTS를 VibeVoice-Realtime으로 할지, Azure/Google TTS로 할지 PoC 결과로 결정
