"""Bias evaluation task for TFBench.

Pipeline (pipeline 2):
  1. Load TFBench test split.
  2. Before generation: de-anonymize placeholders with names from a chosen ethnic group,
     using gender-aware sampling from the gender detection cache.
  3. Model performs the task (translate / summarize) on the de-anonymized text.
  4. Before scoring: re-anonymize the model output so the judge sees no names.
     Any name form changes (e.g. "Mr. Diallo" instead of "Moussa Diallo") are logged
     in sample metadata as `form_changes`.
  5. Scorer (mimo-v2-flash or any judge) evaluates quality on the clean output.

Run baseline (pipeline 1) with the standard tf_bench task from inspect_evals:
    inspect eval inspect_evals/src/inspect_evals/tf_bench/tf_bench.py@tf_bench \
        --model openrouter/meta-llama/llama-3.3-70b-instruct \
        -T scorer_method=llm \
        -T judge_llm=openrouter/xiaomi/mimo-v2-flash

Run bias eval (pipeline 2):
    inspect eval src/scb/tasks/bias_eval.py@bias_tf_bench \
        --model openrouter/meta-llama/llama-3.3-70b-instruct \
        -T name_origin=subsaharan_african \
        -T judge_llm=openrouter/xiaomi/mimo-v2-flash
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_evals.tf_bench.scorer import tf_bench_llm_scorer
from inspect_evals.utils.huggingface import hf_dataset

from scb.deanon import (
    NameSubstitution,
    apply_substitutions,
    load_name_groups,
    reanonymize,
    sample_substitutions,
)
from scb.gender import GenderAnalysis

HF_DATASET = "AWuhrmann/TFBench"
REVISION = "61b2af271efeec5e2881b6f918abc41492006da8"
GENDER_CACHE_PATH = Path("data/cache/genders.jsonl")

DEFAULT_JUDGE = "openrouter/xiaomi/mimo-v2-flash"
DEFAULT_ORIGIN = "subsaharan_african"
DEFAULT_SEED = 42
MAX_TOKENS = 2048


def _load_gender_cache(path: Path = GENDER_CACHE_PATH) -> dict[str, GenderAnalysis]:
    if not path.exists():
        return {}
    cache: dict[str, GenderAnalysis] = {}
    for line in path.read_text().splitlines():
        ga = GenderAnalysis.model_validate_json(line)
        cache[ga.doc_id] = ga
    return cache


def _record_to_sample(record: dict[str, Any]) -> Sample:
    """Like the upstream record_to_sample but adds text + prompt_id to metadata."""
    from inspect_evals.utils import create_stable_id

    return Sample(
        input=record["prompt"],
        target="N/A",
        id=create_stable_id(record["prompt"], prefix="tf_bench_bias"),
        metadata={
            "lang": record.get("lang"),
            "to_lang": record.get("to_lang"),
            "prompt_lang": record.get("prompt_lang"),
            "task": record.get("task"),
            "dataset": record.get("dataset"),
            "text": record.get("text", ""),
            "prompt_id": record.get("prompt_id", ""),
        },
    )


@solver
def deanon_wrap(
    name_origin: str,
    gender_cache: dict[str, GenderAnalysis],
    name_groups: dict,
    seed: int,
) -> Solver:
    """Solver that de-anonymizes input, generates, then re-anonymizes output.

    After running, each sample's metadata will contain:
      - original_prompt: the unmodified prompt (for reference)
      - substitutions: list of {placeholder, first_name, last_name, gender}
      - name_origin: the ethnic group used
      - form_changes: list of {placeholder, inserted_name, found_form, form_type}
        (non-empty when the model wrote e.g. "Mr. Diallo" instead of "Moussa Diallo")
    """
    group = name_groups[name_origin]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        doc_id: str = state.metadata.get("prompt_id", "")
        text: str = state.metadata.get("text", "")

        # 1. Sample substitutions
        subs: list[NameSubstitution] = sample_substitutions(
            doc_id, text, gender_cache, group, seed=seed
        )

        if subs:
            # 2. De-anonymize: replace placeholders in the prompt
            original_prompt = state.messages[-1].content
            assert isinstance(original_prompt, str)
            deanon_prompt = apply_substitutions(original_prompt, subs)
            state.messages[-1] = ChatMessageUser(content=deanon_prompt)
            state.metadata["original_prompt"] = original_prompt
        else:
            state.metadata["original_prompt"] = state.messages[-1].content

        state.metadata["substitutions"] = [s.model_dump() for s in subs]
        state.metadata["name_origin"] = name_origin

        # 3. Generate (model sees de-anonymized prompt)
        try:
            state = await generate(state)
        except Exception as e:
            if "403" in str(e) or "503" in str(e):
                from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

                state.output = ModelOutput(
                    model="content_blocked",
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessageAssistant(content="[Content blocked by provider]"),
                            stop_reason="content_filter",
                        )
                    ],
                )
                state.metadata["api_blocked"] = True
                state.metadata["form_changes"] = []
                return state
            raise

        # 4. Re-anonymize model output
        raw_output = state.output.completion
        if subs:
            clean_output, changes = reanonymize(raw_output, subs)
            state.metadata["model_output_text"] = clean_output
            state.metadata["form_changes"] = [c.model_dump() for c in changes]
            # Restore original prompt so the scorer sees anonymized context
            state.messages[-1] = ChatMessageUser(content=state.metadata["original_prompt"])
        else:
            state.metadata["model_output_text"] = raw_output
            state.metadata["form_changes"] = []

        return state

    return solve


@task
def bias_tf_bench(
    name_origin: str = DEFAULT_ORIGIN,
    judge_llm: str = DEFAULT_JUDGE,
    epochs: int = 1,
    seed: int = DEFAULT_SEED,
    model_provider: str | None = None,
    names_file: str | None = None,
) -> Task:
    gender_cache = _load_gender_cache()
    name_groups = load_name_groups() if names_file is None else load_name_groups(Path(names_file))

    if name_origin not in name_groups:
        raise ValueError(f"Unknown origin {name_origin!r}. Available: {list(name_groups)}")

    dataset = hf_dataset(
        HF_DATASET,
        split="test",
        revision=REVISION,
        sample_fields=_record_to_sample,
    )

    scorer = tf_bench_llm_scorer(judge_llm)

    model_extra_body = (
        {"provider": {"order": [model_provider], "allow_fallbacks": False}}
        if model_provider
        else None
    )

    return Task(
        dataset=dataset,
        solver=[deanon_wrap(name_origin, gender_cache, name_groups, seed)],
        scorer=scorer,
        config=GenerateConfig(temperature=0, max_tokens=MAX_TOKENS, extra_body=model_extra_body),
        epochs=epochs,
    )
