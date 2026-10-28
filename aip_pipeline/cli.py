from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from .common import write_jsonl
from .stages import run_all

ROOT=Path(__file__).resolve().parents[1]
def init():
    sample=ROOT/'sample_input'; sample.mkdir(exist_ok=True)
    write_jsonl(sample/'claims.jsonl',[{"claim_id":"VN-DEMO#c1","document_id":"VN-DEMO","family_id":"VN-DEMO-FAM","priority_date":"2021-01-01","text_vi":"Một hệ thống cảm biến bao gồm bộ xử lý và mô đun truyền dữ liệu.","text_en":"A sensor system includes a processor and a data transmission module."}])
    write_jsonl(sample/'prior_art.jsonl',[{"publication_id":"PA-001","family_id":"PA-FAM-1","publication_date":"2020-01-01","title":"Sensor transmission system","abstract":"A sensor system includes a processor and a data transmission module.","text":"A sensor system includes a processor. The processor communicates through a data transmission module."},{"publication_id":"PA-002","family_id":"PA-FAM-2","publication_date":"2022-01-01","title":"Later publication","abstract":"This must be excluded by date.","text":"A later document."}])
    write_jsonl(sample/'ocr_artifacts.jsonl',[{"claim_id":"VN-DEMO#c1","tesseract_vie_best":"1. Một hệ thống cảm biến bao gồm bộ xử lý và mô đun truyền dữ liệu.","ppocrv6":"1. Một hệ thống cảm biến bao gồm bộ xử lý và mô đun truyền dữ liệu.","tesseract_confidence":0.91,"ppocrv6_confidence":0.88,"tesseract_variants":["1. Một hệ thống cảm biến bao gồm bộ xử lý và mô đun truyền dữ liệu.","1. Một hệ thống cảm biến bao gồm bộ xử lý và mô đun truyền dữ liệu."]}])
    write_jsonl(sample/'features.jsonl',[{"claim_id":"VN-DEMO#c1","method":"demo_provided_external_artifact","features":[{"feature_id":"F1","text":"A sensor system includes a processor and a data transmission module.","start_char":0,"end_char":68,"method":"demo"}]}])
    write_jsonl(sample/'sources.jsonl',[{"source_id":"demo","kind":"local_fixture","license":"demo","accessed_at":"local"}])
    cfg={"input":{"claims":"sample_input/claims.jsonl","prior_art":"sample_input/prior_art.jsonl","sources":"sample_input/sources.jsonl","ocr_artifacts":"sample_input/ocr_artifacts.jsonl","feature_artifacts":"sample_input/features.jsonl"},"output_dir":"runs/latest","retrieval":{"top_k":10,"passages_per_document":3},"features":{"mode":"external_artifacts"},"translation":{"model":"Qwen3-8B-Q4_K_M","model_path":"../models/qwen3-8b-q4_k_m.gguf","n_ctx":16384,"n_gpu_layers":35,"max_tokens":700},"evidence":{"model":"qwen3.5:4b","endpoint":"http://localhost:11434/api/chat","context_length":32768,"max_tokens":512,"timeout":900},"review":{"enabled":false,"model":"gemini-3.5-flash-lite","api_key_env":"GEMINI_API_KEY","endpoint":"https://generativelanguage.googleapis.com/v1beta","top_documents":5,"max_tokens":4096,"timeout":120}}
    (ROOT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding='utf-8')
    print(ROOT/'config.json')
def main():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest='command',required=True); sp.add_parser('init'); r=sp.add_parser('run'); r.add_argument('--config',default='config.json'); a=p.parse_args()
    if a.command=='init': init(); return
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8')); print(json.dumps(run_all(cfg),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
