import json, subprocess, sys, h5py, numpy as np, hashlib
def manifest(backend,path,**kw):
    x={'schema_version':'tdgl-cpu-benchmark-v1','role':'corrected_reference' if backend=='reference' else 'compatible_candidate','backend':backend,'config_hash':'cfg','model_hash':'model','rhs_source_hash':'rhs','julia_version':'1.12','machine':'x','initial_hash':'abc','seed':1,'Je':.18,'noise':0.,'Nx':2,'Ny':2,'hx':.5,'hy':.5,'horizon':2.,'tolerance':1e-4,'snapshot_interval':1.,'skip_ratio':.9,'solver':'Tsit5','precision':'ComplexF64/Float64','save_fields':True,'status':'success','complete':True,'final_time':2.,'wall_seconds':2. if backend=='reference' else 1.,'output_path':str(path),'output_hash':'','initial_vector_path':str(path.with_suffix('.bin')),'initial_vector_hash':'init','output_schema':'streamed_hdf5_fields','rhs_seconds':.1,'accepted_steps':2,'rejected_steps':0,'rhs_calls':10};x.update(kw);return x
def h5(path):
    with h5py.File(path,'w') as f:
        f['times']=[2.];f['V']=[0.];f['psi_avg']=[0.];f['T_avg']=[0.];f['psi_real']=np.zeros((1,3,3));f['psi_imag']=np.zeros((1,3,3));f['T']=np.zeros((1,3,3));f['Ax']=np.zeros((1,3,2));f['Ay']=np.zeros((1,2,3))
def run(tmp,r,c):
    rp,cp=tmp/'r.json',tmp/'c.json';rp.write_text(json.dumps(r));cp.write_text(json.dumps(c));return subprocess.run([sys.executable,'bench/compare_results.py','--reference',str(rp),'--candidate',str(cp),'--out',str(tmp/'x.csv')],capture_output=True)
def test_report_rejects_incompatible_initial_state(tmp_path):
    h5(tmp_path/'r.h5');h5(tmp_path/'c.h5');p=run(tmp_path,manifest('reference',tmp_path/'r.h5'),manifest('cpu_compatible',tmp_path/'c.h5',initial_hash='different'));assert p.returncode==2 and 'mismatch initial_hash' in p.stdout.decode()
def test_report_passes_matched_pair(tmp_path):
    h5(tmp_path/'r.h5');h5(tmp_path/'c.h5');rh=hashlib.sha256((tmp_path/'r.h5').read_bytes()).hexdigest();ch=hashlib.sha256((tmp_path/'c.h5').read_bytes()).hexdigest();p=run(tmp_path,manifest('reference',tmp_path/'r.h5',output_hash=rh),manifest('cpu_compatible',tmp_path/'c.h5',output_hash=ch));assert p.returncode==0 and '"passed": true' in p.stdout.decode()
def test_report_rejects_failed_status(tmp_path):
    h5(tmp_path/'r.h5');h5(tmp_path/'c.h5');p=run(tmp_path,manifest('reference',tmp_path/'r.h5',status='failed'),manifest('cpu_compatible',tmp_path/'c.h5'));assert p.returncode==2 and 'failed status' in p.stdout.decode()
def test_report_rejects_missing_output(tmp_path):
    p=run(tmp_path,manifest('reference',tmp_path/'missing.h5'),manifest('cpu_compatible',tmp_path/'missing2.h5'));assert p.returncode==2 and 'missing output artifact' in p.stdout.decode()
def test_report_rejects_malformed_times(tmp_path):
    h5(tmp_path/'r.h5');h5(tmp_path/'c.h5');
    with h5py.File(tmp_path/'c.h5','a') as f:f['times'][0]=1.
    rh=hashlib.sha256((tmp_path/'r.h5').read_bytes()).hexdigest();ch=hashlib.sha256((tmp_path/'c.h5').read_bytes()).hexdigest();p=run(tmp_path,manifest('reference',tmp_path/'r.h5',output_hash=rh),manifest('cpu_compatible',tmp_path/'c.h5',output_hash=ch));assert p.returncode==2 and 'retained schedule' in p.stdout.decode()
def test_report_rejects_large_numerical_mismatch(tmp_path):
    h5(tmp_path/'r.h5');h5(tmp_path/'c.h5')
    with h5py.File(tmp_path/'c.h5','a') as f:f['V'][0]=100.
    rh=hashlib.sha256((tmp_path/'r.h5').read_bytes()).hexdigest();ch=hashlib.sha256((tmp_path/'c.h5').read_bytes()).hexdigest();p=run(tmp_path,manifest('reference',tmp_path/'r.h5',output_hash=rh),manifest('cpu_compatible',tmp_path/'c.h5',output_hash=ch));assert p.returncode==2 and 'field error exceeds' in p.stdout.decode()

def test_report_accepts_final_fractional_snapshot(tmp_path):
    for name in ('r.h5','c.h5'):
        h5(tmp_path/name)
        with h5py.File(tmp_path/name,'a') as f:f['times'][0]=2.5
    rh=hashlib.sha256((tmp_path/'r.h5').read_bytes()).hexdigest();ch=hashlib.sha256((tmp_path/'c.h5').read_bytes()).hexdigest()
    p=run(tmp_path,manifest('reference',tmp_path/'r.h5',output_hash=rh,horizon=2.5,final_time=2.5),manifest('cpu_compatible',tmp_path/'c.h5',output_hash=ch,horizon=2.5,final_time=2.5))
    assert p.returncode==0,p.stdout.decode()

def test_report_rejects_observable_dimension_mismatch(tmp_path):
    h5(tmp_path/'r.h5');h5(tmp_path/'c.h5')
    with h5py.File(tmp_path/'c.h5','a') as f:
        del f['V'];f['V']=np.zeros((1,1))
    rh=hashlib.sha256((tmp_path/'r.h5').read_bytes()).hexdigest();ch=hashlib.sha256((tmp_path/'c.h5').read_bytes()).hexdigest()
    p=run(tmp_path,manifest('reference',tmp_path/'r.h5',output_hash=rh),manifest('cpu_compatible',tmp_path/'c.h5',output_hash=ch))
    assert p.returncode==2 and 'dimension mismatch V' in p.stdout.decode()
