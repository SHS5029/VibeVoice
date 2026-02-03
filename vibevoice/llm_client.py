"""VibeVoice LLM Client - GLM-4 & OpenAI API 통합"""
import os
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, List, Dict, Optional


class LLMClient:
    """통화 분석을 위한 LLM 클라이언트 (GLM-4 & OpenAI 지원)"""
    
    def __init__(
        self, 
        provider: str = "glm",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 30
    ):
        """
        Args:
            provider: "glm" (GLM-4) 또는 "openai"
            api_key: API 키 (None이면 환경변수에서 자동 로드)
            model: 모델 이름 (None이면 provider별 기본값 사용)
            base_url: API 엔드포인트 (GLM-4 커스텀 엔드포인트용)
        """
        self.provider = provider.lower()
        self.client: Optional[Any] = None
        self.timeout_seconds = timeout_seconds
        self.has_openai = False
        self._openai_class = None

        try:
            openai_module = __import__("openai")
            self._openai_class = getattr(openai_module, "OpenAI", None)
            if self._openai_class is not None:
                self.has_openai = True
        except Exception:
            self.has_openai = False
        
        # API 키 자동 로드
        if api_key is None:
            if self.provider == "glm":
                api_key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("GLM_API_KEY")
            elif self.provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
        
        self.api_key = api_key
        
        # 모델 기본값 설정
        if model is None:
            if self.provider == "glm":
                self.model = "glm-4.7-flash"  # 무료 티어
            elif self.provider == "openai":
                self.model = "gpt-4o-mini"
            else:
                self.model = "glm-4.7-flash"
        else:
            self.model = model
        
        # Base URL 설정
        if base_url is None and self.provider == "glm":
            base_url = "https://api.z.ai/api/paas/v4/"
        
        # OpenAI SDK로 클라이언트 초기화 (GLM-4도 OpenAI 호환)
        if self.has_openai and self.api_key and self._openai_class is not None:
            try:
                if base_url:
                    self.client = self._openai_class(api_key=self.api_key, base_url=base_url)
                else:
                    self.client = self._openai_class(api_key=self.api_key)
            except Exception as e:
                print(f"LLM 클라이언트 초기화 실패: {e}")
                self.client = None
    
    def analyze_call(self, transcription_segments: List[Dict]) -> Dict:
        """통화 내용 분석"""
        if not transcription_segments:
            return self._empty_result()
        
        dialogue = self._build_dialogue(transcription_segments)
        
        # API 호출 가능하면 실제 분석
        if self.client:
            return self._analyze_with_api(dialogue)
        
        # 아니면 규칙 기반 분석
        return self._rule_based_analysis(dialogue)
    
    def _build_dialogue(self, segments: List[Dict]) -> str:
        """세그먼트를 대화 텍스트로 변환"""
        lines = []
        for seg in segments:
            speaker = f"화자 {seg.get('speaker_id', '?')}"
            text = seg.get('text', '')
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)
    
    def _analyze_with_api(self, dialogue: str) -> Dict:
        """LLM API로 분석 (GLM-4 또는 OpenAI)"""
        prompt = f"""다음 콜센터 통화 내용을 분석하세요.

통화 내용:
{dialogue}

다음 JSON 형식으로 응답하세요:
{{
    "summary": "2-3문장 요약",
    "situation": "현재 상황 설명",
    "intent": "고객 의도 (문의/불만/요청/칭찬 중 하나)",
    "sentiment": "감정 상태 (긍정/중립/부정 중 하나)",
    "next_actions": ["권장 조치 1", "권장 조치 2"]
}}"""

        content = ""
        try:
            response = self._call_with_timeout(prompt)
            
            content = response.choices[0].message.content.strip()
            
            # JSON 파싱 (코드 블록 제거)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            result = json.loads(content)
            return {
                "summary": result.get("summary", "분석 결과 없음"),
                "situation": result.get("situation", "알 수 없음"),
                "intent": result.get("intent", "알 수 없음"),
                "sentiment": result.get("sentiment", "중립"),
                "next_actions": result.get("next_actions", [])
            }
            
        except FuturesTimeoutError:
            print(f"LLM API 타임아웃: {self.timeout_seconds}초 초과")
            return self._rule_based_analysis(dialogue)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}\n응답: {content}")
            return self._rule_based_analysis(dialogue)
        except Exception as e:
            print(f"LLM API 오류: {e}")
            return self._rule_based_analysis(dialogue)

    def _call_with_timeout(self, prompt: str):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._create_completion, prompt)
            return future.result(timeout=self.timeout_seconds)

    def _create_completion(self, prompt: str):
        if self.client is None:
            raise RuntimeError("LLM client not initialized")
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 콜센터 통화 분석 전문가입니다. 한국어로 응답하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
    
    def _rule_based_analysis(self, dialogue: str) -> Dict:
        """규칙 기반 분석 (API 없을 때)"""
        lower = dialogue.lower()
        
        # 의도 분류
        intent = "일반 문의"
        if any(w in lower for w in ["주문", "배송", "언제 오"]):
            intent = "배송/주문 문의"
        elif any(w in lower for w in ["환불", "반품", "취소"]):
            intent = "환불/반품 요청"
        elif any(w in lower for w in ["고장", "안돼", "오류", "에러"]):
            intent = "기술 지원/AS"
        elif any(w in lower for w in ["변경", "수정", "바꿔"]):
            intent = "정보 변경 요청"
        
        # 감정 분류
        sentiment = "중립"
        if any(w in lower for w in ["감사", "좋아", "잘", "친절"]):
            sentiment = "긍정"
        elif any(w in lower for w in ["화나", "불편", "짜증", "언제까지", "빨리"]):
            sentiment = "부정"
        
        # 다음 조치
        actions = ["고객 정보 확인", "상세 내역 조회"]
        if "환불" in lower or "반품" in lower:
            actions = ["환불/반품 절차 안내", "관련 부서 연결"]
        elif "고장" in lower or "안돼" in lower:
            actions = ["문제 상황 상세 확인", "기술 지원팀 연결"]
        
        # API 상태 메시지
        api_note = ""
        if not self.client:
            if not self.has_openai:
                api_note = " (규칙 기반 - pip install openai 필요)"
            elif not self.api_key:
                if self.provider == "glm":
                    api_note = " (규칙 기반 - ZHIPUAI_API_KEY 환경변수 설정 필요)"
                else:
                    api_note = " (규칙 기반 - OPENAI_API_KEY 환경변수 설정 필요)"
        
        return {
            "summary": f"고객 문의가 접수되었습니다.{api_note}",
            "situation": "고객과 상담원 간 통화 진행 중",
            "intent": intent,
            "sentiment": sentiment,
            "next_actions": actions
        }
    
    def _empty_result(self) -> Dict:
        return {
            "summary": "대화 내용이 없습니다.",
            "situation": "알 수 없음",
            "intent": "알 수 없음",
            "sentiment": "중립",
            "next_actions": []
        }
