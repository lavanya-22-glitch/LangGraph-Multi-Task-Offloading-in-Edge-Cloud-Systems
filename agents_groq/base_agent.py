from groq import Groq
import re

class BaseAgent:
    """Base class for all Groq-powered agents with Chain-of-Thought support."""

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def _extract_content(self, response):
        """
        Extract text content from LLM response, handling both string and list formats.
        
        Args:
            response: Groq response object
            
        Returns:
            String content
        """
        # For Groq API, response is typically response.choices[0].message.content
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError):
            # Fallback if structure is different
            content = str(response)
        
        # Handle list of content blocks (multimodal responses)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    # Handle dict format: {"type": "text", "text": "..."}
                    if block.get("type") == "text" and "text" in block:
                        text_parts.append(block["text"])
                    # Handle other dict formats
                    elif "text" in block:
                        text_parts.append(block["text"])
                elif isinstance(block, str):
                    # Handle string blocks
                    text_parts.append(block)
                elif hasattr(block, 'text'):
                    # Handle objects with text attribute
                    text_parts.append(block.text)
            content = "\n".join(text_parts)
        
        # Ensure we have a string
        if not isinstance(content, str):
            content = str(content)
        
        return content.strip()

    def think(self, prompt: str):
        """Run LLM reasoning and return response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        return self._extract_content(response)

    def think_with_cot(self, prompt: str, return_reasoning: bool = False):
        """
        Run LLM reasoning with Chain-of-Thought approach.
        
        Args:
            prompt: The input prompt
            return_reasoning: If True, returns both reasoning and answer
            
        Returns:
            If return_reasoning=False: Just the final answer
            If return_reasoning=True: Dict with 'reasoning' and 'answer'
        """
        cot_prompt = f"""
{prompt}

Think through this step-by-step:
1. First, analyze the problem and identify key constraints
2. Consider different approaches and their trade-offs
3. Reason through the implications of each decision
4. Arrive at a well-justified conclusion

IMPORTANT: Format your response EXACTLY as shown below, with opening and closing tags on their own lines:

<reasoning>
[Your detailed step-by-step thinking process here]
</reasoning>

<answer>
[Your final answer here]
</answer>

Do not include any text before <reasoning> or after </answer>.
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": cot_prompt}
            ],
            temperature=self.temperature,
        )
        content = self._extract_content(response)
        
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else content
        
        if return_reasoning:
            return {
                "reasoning": reasoning,
                "answer": answer
            }
        return answer

    def think_with_self_consistency(self, prompt: str, num_samples: int = 3):
        """
        Use self-consistency: generate multiple reasoning paths and select most consistent answer.
        
        Args:
            prompt: The input prompt
            num_samples: Number of reasoning paths to generate
            
        Returns:
            The most consistent answer with reasoning
        """
        responses = []
        
        for i in range(num_samples):
            cot_prompt = f"""
{prompt}

Think through this step-by-step. This is reasoning path {i+1}/{num_samples}.
Show your work and explain your reasoning clearly.

IMPORTANT: Format your response EXACTLY as shown below:

<reasoning>
[Your step-by-step thinking]
</reasoning>

<answer>
[Your final answer]
</answer>
"""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": cot_prompt}
                ],
                temperature=self.temperature,
            )
            content = self._extract_content(response)
            
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            
            if reasoning_match and answer_match:
                responses.append({
                    "reasoning": reasoning_match.group(1).strip(),
                    "answer": answer_match.group(1).strip()
                })
        
        return responses[0] if responses else {"reasoning": "", "answer": ""}