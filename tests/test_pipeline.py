import json, tempfile, unittest
from pathlib import Path
from aip_pipeline.stages import run_all
from aip_pipeline.ocr_hybrid import process

class PipelineTests(unittest.TestCase):
 def test_end_to_end_filters_date_and_family(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); claims=root/'claims.jsonl'; prior=root/'prior.jsonl'; sources=root/'sources.jsonl'
   claims.write_text(json.dumps({'claim_id':'c1','document_id':'d1','family_id':'self','priority_date':'2021-01-01','text_vi':'sensor processor data transmission module','text_en':'sensor processor data transmission module'})+'\n',encoding='utf8')
   prior.write_text('\n'.join(json.dumps(x) for x in [{'publication_id':'selfpub','family_id':'self','publication_date':'2020-01-01','text':'sensor processor data transmission module'},{'publication_id':'later','family_id':'other','publication_date':'2022-01-01','text':'sensor processor data transmission module'},{'publication_id':'good','family_id':'other','publication_date':'2020-01-01','text':'sensor processor data transmission module'}])+'\n',encoding='utf8')
   sources.write_text('',encoding='utf8'); out=root/'out'; run=run_all({'input':{'claims':str(claims),'prior_art':str(prior),'sources':str(sources)},'output_dir':str(out),'features':{'mode':'rule'},'retrieval':{'top_k':10,'passages_per_document':1},'evidence':{'model':'lexical'}})
   self.assertEqual(run['status'],'COMPLETED'); row=json.loads((out/'05_retrieval.jsonl').read_text(encoding='utf8')); self.assertEqual([x['publication_id'] for x in row['candidate_documents']],['good']); self.assertTrue((out/'07_evidence.jsonl').exists()); review=[json.loads(x) for x in (out/'08_reviewer_report.jsonl').read_text(encoding='utf8').splitlines()]; self.assertEqual(review[0]['status'],'not_run')
 def test_hybrid_ocr_repairs_only_consensus_marker(self):
  row=process({'tesseract_vie_best':'1. Bộ cảm biến\nBộ truyền dữ liệu','ppocrv6':'1. Bộ cảm biến\nBộ truyền dữ liệu','tesseract_variants':['1. Bộ cảm biến\n2. Bộ truyền dữ liệu','1. Bộ cảm biến\n2. Bộ truyền dữ liệu']})
  self.assertIn('2. Bộ truyền dữ liệu',row['text_vi']); self.assertEqual(row['marker_repairs'][0]['method'],'consensus_marker_repair_v6_v7')
if __name__=='__main__': unittest.main()
