import argparse
import json
from pathlib import Path
from . import validate

def main():
    parser=argparse.ArgumentParser(description='Validate a Railbound JSON solution')
    parser.add_argument('level',type=Path,nargs='?')
    parser.add_argument('solution',type=Path,nargs='?')
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--dataset',type=Path,nargs='?',const=root/'levels/extended-hard')
    parser.add_argument('--solutions',type=Path)
    parser.add_argument('--history-steps',type=int)
    parser.add_argument('--history-gap',type=int,default=1)
    parser.add_argument('--max-ticks',type=int,default=10000)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.dataset is not None:
        if args.level or args.solution:parser.error('Use --dataset without positional files')
        from .dataset import validate_dataset
        solutions = args.solutions or root/'solutions'/args.dataset.name
        result=validate_dataset(args.dataset,solutions,args.max_ticks)
    else:
        if args.level is None or args.solution is None:parser.error('Provide a level and solution, or --dataset')
        result=validate(args.level,args.solution,args.history_steps,args.history_gap,args.max_ticks)
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('trace','feedback_states')}))
    else: print(text)
    return 0 if result['success'] else 1

if __name__=='__main__': raise SystemExit(main())
