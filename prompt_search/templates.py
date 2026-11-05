"""Active generation and aggregation templates from notebook cell 6."""
import textwrap
from typing import List

class ManualMicrobiomePrompter:
    TASK_DESCRIPTION = 'I have extracted the **Top 50 most abundant OTUs** (Operational Taxonomic Units) from a human gut microbiome sample, along with their **relative abundances**.\nYour task is to identify **functional collaboration groups (FCGs)** — small sets of OTUs.'
    OUTPUT_FORMAT = '### Output Format (Markdown)\n### Functional Collaboration Groups\n- **Group 1**: OTU_001, OTU_007, OTU_014  \n**Function**: ...  \n- **Group 2**: OTU_023, OTU_034, OTU_056, OTU_061  \n**Function**: ...'

    def __init__(self, dataset_hint: str='', hard_limits: str=''):
        self.dataset_hint = dataset_hint
        self.hard_limits = hard_limits

    def generate_prompt_template(self) -> str:
        output_format_section = f'\nREQUIRED OUTPUT FORMAT (must keep EXACTLY as specified):\n-------------------------------------------------------\n{self.OUTPUT_FORMAT}\n'
        return textwrap.dedent(f'\n        You are a senior prompt engineer. Create a new prompt to maximize downstream validation accuracy for identifying functional collaboration groups (FCGs) in gut microbiome data.\n\n        TASK DESCRIPTION (reference):\n        -----------------------------\n        {self.TASK_DESCRIPTION}\n\n        {output_format_section}\n\n        STRICT REQUIREMENTS:\n        -------------------\n        1) You MUST keep the output format EXACTLY as specified above (no changes to the structure).\n        2) The task LLM will receive an OTU list via a placeholder {{otu_text}} (keep this placeholder in your final prompt).\n        3) make sure your final prompt explicitly instructs the task LLM to generate at least 10 groups.\n        4) Keep ASCII-friendly punctuation and stable, compact wording in **Function** lines (≤ 1–2 sentences).\n\n        Return ONLY the complete prompt text (no explanations).\n        ').strip()

    def aggregation_prompt_template(self, prompts: List[str], scores: List[float | None]) -> str:
        assert len(prompts) >= 2, '聚合至少需要两个 prompt'
        prompts_with_scores = []
        for i, (p, score) in enumerate(zip(prompts, scores)):
            if score is None:
                score_info = '(Score: N/A)'
            elif score >= 0:
                score_info = f'(Score: {score:.4f})'
            else:
                score_info = '(INVALID - failed to generate NPZ)'
            prompts_with_scores.append(f'Prompt {chr(65 + i)} {score_info}:\n{p}')
        prompts_text = '\n\n'.join(prompts_with_scores)
        output_format_section = f'\nREQUIRED OUTPUT FORMAT (must keep EXACTLY as specified):\n-------------------------------------------------------\n{self.OUTPUT_FORMAT}\n'
        return textwrap.dedent(f'\n        You are a senior prompt engineer. Fuse the following {len(prompts)} prompts into a single improved prompt to maximize downstream validation accuracy for identifying functional collaboration groups (FCGs) in gut microbiome data, WITHOUT changing the output format.\n\n        {output_format_section}\n\n        Return ONLY the single fused prompt text (no explanations).\n\n        PROMPTS TO FUSE:\n        ---------------\n        {prompts_text}\n\n        Return ONLY the fused prompt (no explanations).\n        ').strip()
