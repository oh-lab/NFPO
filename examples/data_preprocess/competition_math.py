# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Competition MATH dataset to parquet format.
"""

import argparse
from collections import Counter
import json
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


DATA_SOURCE = "qwedsacf/competition_math"
DEFAULT_SUBJECTS = [
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
]
SUBJECT_NAME_ALIASES = {
    "algebra": "Algebra",
    "counting and probability": "Counting & Probability",
    "geometry": "Geometry",
    "intermediate algebra": "Intermediate Algebra",
    "number theory": "Number Theory",
    "prealgebra": "Prealgebra",
    "precalculus": "Precalculus",
}
INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."


def parse_levels(levels_arg: str) -> list[int]:
    levels = []
    for token in levels_arg.split(","):
        token = token.strip()
        if not token:
            continue
        level = int(token)
        if level < 1 or level > 5:
            raise ValueError(f"Invalid level {level}. Expected values between 1 and 5.")
        levels.append(level)
    if not levels:
        raise ValueError("At least one level must be provided.")
    return sorted(set(levels))


def normalize_level(level_value, *, strict: bool = False) -> int | None:
    if isinstance(level_value, int):
        return level_value

    if isinstance(level_value, str):
        match = re.search(r"(\d+)", level_value)
        if match is not None:
            return int(match.group(1))

    if strict:
        raise ValueError(f"Could not parse level value: {level_value!r}")

    return None


def normalize_subject_name(subject_name: str | None) -> str:
    if subject_name is None:
        return "Unknown"

    normalized = subject_name.strip().lower().replace("&", "and").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return SUBJECT_NAME_ALIASES.get(normalized, subject_name.strip())


def extract_answer(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return solution.strip()

    answer = remove_boxed(boxed)
    if answer is None:
        return solution.strip()

    return answer.strip()


def subject_display_name(example: dict) -> str:
    subject_name = normalize_subject_name(example.get("type"))
    if subject_name != "Unknown":
        return subject_name

    return "Competition Math"


def subject_config_name(example: dict) -> str:
    return (example.get("type") or "competition_math").strip().lower().replace(" ", "_")


def load_raw_dataset(data_path: str, preferred_split: str):
    try:
        return datasets.load_dataset(data_path, split=preferred_split), preferred_split
    except Exception as split_error:
        dataset_bundle = datasets.load_dataset(data_path)

        if isinstance(dataset_bundle, datasets.Dataset):
            print(
                f"Dataset {data_path} did not expose named splits; using the loaded dataset directly.",
                flush=True,
            )
            return dataset_bundle, preferred_split or "default"

        if preferred_split in dataset_bundle:
            return dataset_bundle[preferred_split], preferred_split

        if len(dataset_bundle) == 1:
            split_name = next(iter(dataset_bundle.keys()))
            print(
                f"Preferred split {preferred_split!r} was unavailable; using the only available split {split_name!r}.",
                flush=True,
            )
            return dataset_bundle[split_name], split_name

        available_splits = ", ".join(dataset_bundle.keys())
        raise ValueError(
            f"Could not load split {preferred_split!r} from {data_path}. "
            f"Available splits: {available_splits}"
        ) from split_error


def make_map_fn(split: str, dataset_source: str):
    def process_fn(example, idx):
        question = example["problem"].strip()
        question = question + " " + INSTRUCTION_FOLLOWING

        solution = example["solution"]
        level = normalize_level(example["level"], strict=True)
        subject_name = subject_display_name(example)
        subject_config = subject_config_name(example)
        unique_id = example.get("unique_id") or example.get("file") or f"{split}/{subject_config}/{idx}"

        return {
            "subject": subject_name,
            "level": level,
            "unique_id": unique_id,
            "data_source": dataset_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": extract_answer(solution)},
            "extra_info": {
                "split": split,
                "index": idx,
                "subject_config": subject_config,
                "subject_type": example.get("type"),
                "original_level": example["level"],
                "solution": solution,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--dataset_source",
        default=DATA_SOURCE,
        help="The Hugging Face dataset repo to load when local_dataset_path is not provided.",
    )
    parser.add_argument(
        "--dataset_split",
        default="train",
        help="Preferred split name. Falls back to the only available split when needed.",
    )
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/competition_math_level3to5",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--levels",
        default="3,4,5",
        help="Comma-separated list of difficulty levels to keep. Example: 3,4,5",
    )
    parser.add_argument(
        "--subjects",
        default=",".join(DEFAULT_SUBJECTS),
        help="Comma-separated list of subject names to keep.",
    )

    args = parser.parse_args()

    allowed_levels = set(parse_levels(args.levels))
    subject_names = {normalize_subject_name(subject) for subject in args.subjects.split(",") if subject.strip()}
    if not subject_names:
        raise ValueError("At least one subject name must be provided.")

    dataset_source = args.dataset_source
    data_path = args.local_dataset_path or dataset_source
    print(f"Loading the {dataset_source} dataset from {data_path}...", flush=True)

    raw_train_dataset, split_name = load_raw_dataset(data_path, args.dataset_split)
    before_count = len(raw_train_dataset)
    filtered_train_dataset = raw_train_dataset.filter(
        lambda example: normalize_level(example["level"]) in allowed_levels
        and normalize_subject_name(example.get("type")) in subject_names
    )
    after_count = len(filtered_train_dataset)

    if after_count == 0:
        raise ValueError(
            f"No examples remained after filtering split={split_name!r}, "
            f"levels={sorted(allowed_levels)}, subjects={sorted(subject_names)}"
        )

    subject_counts = Counter(normalize_subject_name(example.get("type")) for example in filtered_train_dataset)
    subject_summary = ", ".join(f"{subject}={subject_counts[subject]}" for subject in sorted(subject_counts))
    print(
        f"Loaded split={split_name}: kept {after_count}/{before_count} examples for levels "
        f"{sorted(allowed_levels)} and subjects {sorted(subject_names)}.",
        flush=True,
    )
    print(f"Subject counts: {subject_summary}", flush=True)

    train_dataset = filtered_train_dataset.map(
        function=make_map_fn(split_name, dataset_source),
        with_indices=True,
        remove_columns=filtered_train_dataset.column_names,
    )

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
