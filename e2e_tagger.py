import sys
with open('.env') as f:
    for line in f:
        key, value = line.strip().split('=')
        sys.path.append(value)

import asyncio
import argparse
import json
import yaml

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prediction import get_predictions
from utils import sigmoid

from danbooru.fetch_tags import main as fetch_danbooru_tag
from danbooru.fetch_tags import known_img_suffix, img_dirwalk


@dataclass
class TaggerConfig:
    keep_catagories: tuple[str] = ('general',)  # ('general', 'character', 'copyright', 'artist', 'meta')
    tagger_threshold1: float = 0.3
    tagger_threshold2: float = 0.55
    model_path: str | Path = 'src/Augmented-DDTagger/models/wd-v1-4-swinv2-tagger-v2'
    combine_mode: str = 'AND'
    fallback: bool = True

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self._keep_catagories = set(self.keep_catagories)


raw_tags = np.loadtxt('src/Augmented-DDTagger/models/wd-v1-4-swinv2-tagger-v2/selected_tags.csv', dtype=str, delimiter=',')[1:]

all_tags = raw_tags[:, 1].tolist()
count = raw_tags[:, 3].astype(np.int32)
category = raw_tags[:, 2].astype(np.int32)


MASK = np.zeros_like(count, dtype=bool)
mask_general = (category == 0)
mask_character = (category == 4)
mask_meta = (category == 9)


def get_danbooru_tags_dict(img_path: Path) -> dict[str, list[str]] | None:
    if img_path.with_suffix('.json').exists():
        with open(img_path.with_suffix('.json')) as f:
            db_json = json.load(f)
        tags_dict = dict(
            general=db_json['tag_string_general'].split(),
            character=db_json['tag_string_character'].split(),
            copyright=db_json['tag_string_copyright'].split(),
            artist=db_json['tag_string_artist'].split(),
            meta=db_json['tag_string_meta'].split(),
        )
    else:
        tags_dict = None
    return tags_dict


def get_pred_dict(img_path: Path, score: np.ndarray, tags: list[str]) -> dict[str, np.float32]:
    mask = MASK & (score > 0.001)
    indices = np.where(mask)[0]
    order = np.argsort(score[indices])[::-1]
    return {tags[i]: score[i] for i in indices[order]}


def get_original_label(img_path: Path) -> str | None:
    if img_path.with_suffix('.txt').exists():
        with open(img_path.with_suffix('.txt')) as f:
            return f.read().strip()


def combine_dd_first(config: TaggerConfig, danbooru_tags: dict[str, str] | None, tagger_preds: dict[str, float], original_label: str):
    if danbooru_tags:
        return [tag for cate, taglist in danbooru_tags.items() for tag in taglist if cate in config._keep_catagories]
    elif not config.fallback:
        raise ValueError('DD_ONLY is True but no danbooru tags found')
    return [tag for tag, confid in tagger_preds.items() if confid > config.tagger_threshold2]


def combine_AND(config: TaggerConfig, danbooru_tags: dict[str, str] | None, tagger_preds: dict[str, float], original_label: str):
    """
    Args:
        danbooru_tags: category [general, meta, character, artist] -> tag with _
        tagger_preds: tag -> confidence. Sorted by confidence in descending
        original_tags: original tags
    """
    if danbooru_tags:
        dd_tags = set([tag for cate, taglist in danbooru_tags.items() for tag in taglist if cate in config._keep_catagories])
        return_tags = []
        for tag, confid in tagger_preds.items():
            if confid < config.tagger_threshold1:
                break
            if tag in dd_tags:
                return_tags.append(tag)
        return return_tags
    return [tag for tag, confid in tagger_preds.items() if confid > config.tagger_threshold2]


def combine_OR(config: TaggerConfig, danbooru_tags: dict[str, str] | None, tagger_preds: dict[str, float], original_label: str):
    tags = [tag for tag, confid in tagger_preds.items() if confid > config.tagger_threshold2]
    if danbooru_tags:
        dd_tags = set([tag for cate, taglist in danbooru_tags.items() for tag in taglist if cate in config._keep_catagories])
        tags = list(dd_tags | set(tags))
    return tags


def combine_tagger_first(config: TaggerConfig, danbooru_tags: dict[str, str] | None, tagger_preds: dict[str, float], original_label: str):
    return [tag for tag, confid in tagger_preds.items() if confid > config.tagger_threshold2]


def combine(config: TaggerConfig, pred_dict: dict[str, np.float32], danbooru_tag_dict: dict[str, list[str]] | None, original_label: str | None) -> list[str]:
    assert config.combine_mode in ('AND', 'OR', 'DD-first', 'TAGGER-first')

    ret_dict = {
        'AND': combine_AND,
        'OR': combine_OR,
        'DD-first': combine_dd_first,
        'TAGGER-first': combine_tagger_first,
    }
    return ret_dict[config.combine_mode](config, danbooru_tag_dict, pred_dict, original_label)  # type: ignore


def an_astolfo_is_1girl(all_tags: list[str]):
    """Generate gender related tags according to the tagger's prediction. If it looks like 1girl, then it is 1girl."""

    gp1 = ['1girl', '2girls', '3girls', '4girls', '5girls', '6+girls']
    gp2 = ['1boy', '2boys', '3boys', '4boys', '5boys', '6+boys']
    gp3 = ['no_humans']
    rm = ['genderswap', 'genderswap_(mtf)', 'genderswap_(ftm)', 'ambiguous_gender'] + gp1 + gp2
    rm = set(rm)
    groups = [gp1, gp2, gp3]
    gp_indices = [np.array([all_tags.index(t) for t in g]) for g in groups]

    def make_astolfo_girl(config: TaggerConfig, combined_tags: list[str], score: np.ndarray):
        gp_scores = [score[idxs] for idxs in gp_indices]
        gp_scores_max = [gp_score.max() for gp_score in gp_scores]
        top_gp = np.argmax(gp_scores_max)
        top_loc = np.argmax(score[gp_indices[top_gp]])
        keep = [groups[top_gp][top_loc]]
        for i in range(len(groups)):
            if i == top_gp:
                continue
            if gp_scores_max[i] > config.tagger_threshold2:
                keep.append(groups[i][np.argmax(gp_scores[i])])
        for tag in combined_tags:
            if tag not in rm:
                keep.append(tag)
        return keep
    return make_astolfo_girl


useless_tags = set(
    ['virtual_youtuber'] +
    [tag for tag in all_tags if 'alternate_' in tag] +
    ['genderswap', 'genderswap_(mtf)', 'genderswap_(ftm)', 'ambiguous_gender']
)


def rm_useless_tags(combined_tags):
    return [tag for tag in combined_tags if tag not in useless_tags]


make_astolfo_girl = an_astolfo_is_1girl(all_tags)


def process(
    config: TaggerConfig,
    pred_dict: dict[str, np.float32],
    scores: np.ndarray,
    danbooru_tag_dict: dict[str, list[str]] | None,
    original_label: str | None,
):
    tags = combine(config, pred_dict, danbooru_tag_dict, original_label)
    tags = make_astolfo_girl(config, tags, scores)
    tags = rm_useless_tags(tags)
    return tags


class Postprocess:
    def __init__(
        self,
        rm: list[str] | None = None,
        replace: list[tuple[str, str]] | None = None,
        prepend: list[str] | None = None,
        append: list[str] | None = None,
        must_have: list[str] | None = None,
    ):
        self.rm = set(rm) if rm else set()
        self.replace = {k: v for k, v in replace} if replace else {}
        _prepend = set(prepend) if prepend else set()
        self.append = list(set(append)) if append else []
        if must_have:
            for t in must_have:
                self.rm.add(t)
                _prepend.add(t)
        self.prepend = list(_prepend)

    def __call__(self, tags: list[str]) -> list[str]:
        filtered_tags = []
        for t in tags:
            if t in self.rm:
                continue
            if t in self.replace:
                t = self.replace[t]
            filtered_tags.append(t)
        return self.prepend + filtered_tags + self.append

    def update(self, p: Path) -> 'Postprocess':
        if not p.exists():
            return deepcopy(self)
        with open(p, 'r') as f:
            other: dict = yaml.load(f, Loader=yaml.FullLoader)
        if not other:
            return deepcopy(self)
        for k, v in other.items():
            if v.__class__ is not list:
                other[k] = [v]

        rm = list(self.rm) + other.get('rm', [])
        replace = list(self.replace.items()) + [tuple(v) for v in other.get('replace', [])]
        prepend = list(self.prepend) + other.get('prepend', [])
        append = list(self.append) + other.get('append', [])
        must_have = list(other.get('must_have', []))
        return Postprocess(rm, replace, prepend, append, must_have)


def _img_dirwalk_with_postprocessor(path: Path, postprocessor: Postprocess):

    postprocessor = postprocessor.update(path / 'rules.yaml')
    if not path.name.startswith('_'):
        name = path.name.split('#', 1)[0].strip()
        postprocessor.prepend.append(name)

    for p in path.iterdir():
        if p.is_dir():
            yield from _img_dirwalk_with_postprocessor(p, postprocessor)
        else:
            if p.suffix[1:] in known_img_suffix:
                yield p, postprocessor


def img_dirwalk_with_postprocessor(path: Path | str):
    path = Path(path)
    postprocess = Postprocess()
    yield from _img_dirwalk_with_postprocessor(path, postprocess)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=False, default='/tmp/momoko', help='Path to image or directory')
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path to model for AugDD')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference in AugDD')
    parser.add_argument('--backend', type=str, default='WD14-SwinV2', help='Backend model to use in AugDD', choices=['WD14-SwinV2', 'WD14-ConvNext', 'DeepDanbooru', 'WD14'])
    parser.add_argument('--nproc', type=int, default=-1, help='Number of processes to use for AugDD. -1 means all')
    parser.add_argument('--max_chunk', type=int, default=16, help='Maximum number of batches to process before one save in AugDD')
    parser.add_argument('--danbooru-concurrency', type=int, default=2, help='Number of concurrent requests to Danbooru')
    parser.add_argument('--iqdb-concurrency', type=int, default=2, help='Number of concurrent requests to iqdb')
    parser.add_argument('--retry-nomatch', action='store_true', help='Retry images that have no matches on iqdb')
    parser.add_argument('--keep-catagories', nargs='+', default=['general'], help='Available: general, character, artist, copyright, meta')
    parser.add_argument('--threshold1', '-t1', type=float, default=0.3, help='Tagger threshold for AND mode')
    parser.add_argument('--threshold2', '-t2', type=float, default=0.55, help='Tagger threshold for OR and fallback mode')
    parser.add_argument('--combine-mode', '-cm', type=str, default='AND', choices=['AND', 'OR', 'TAGGER-first', 'DD-first'])
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    if not args.model_path:
        args.model_path = 'src/Augmented-DDTagger/models/wd-v1-4-swinv2-tagger-v2'
        print('Using default model path:', args.model_path)

    if 'general' in args.keep_catagories:
        MASK |= mask_general
    if 'character' in args.keep_catagories:
        MASK |= mask_character
    if 'meta' in args.keep_catagories:
        MASK |= mask_meta

    config = TaggerConfig(
        keep_catagories=tuple(args.keep_catagories),
        tagger_threshold1=args.threshold1,
        tagger_threshold2=args.threshold2,
        model_path=args.model_path,
        combine_mode=args.combine_mode,
        fallback=True,
    )

    path = Path(args.path)

    print('Running AugDD...')
    scores, rel_paths = get_predictions(
        model_path=args.model_path,
        root_path=path,
        batch_size=args.batch_size,
        backend=args.backend,
        nproc=args.nproc,
        max_chunk=args.max_chunk,
    )
    scores = sigmoid(np.float32(scores))

    score_dicts = {k: v for k, v in zip(rel_paths, scores)}  # type: ignore $ img_path -> confidence

    print('Fetching Danbooru tags...')
    asyncio.run(fetch_danbooru_tag(args.path, args.iqdb_concurrency, args.danbooru_concurrency, args.retry_nomatch))

    for img_path, postproc in img_dirwalk_with_postprocessor(path):
        danbooru_tag_dict = get_danbooru_tags_dict(img_path)
        scores = score_dicts[str(img_path.relative_to(path))]
        pred_dict = get_pred_dict(img_path, scores, all_tags)
        original_label = get_original_label(img_path)
        tags = process(config, pred_dict, scores, danbooru_tag_dict, original_label)
        tags = postproc(tags)
        with open(img_path.with_suffix('.txt'), 'w') as f:
            f.write(' '.join(tags))
