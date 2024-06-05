import streamlit as st
from streamlit_extras.stateful_button import button
import pandas as pd
import time
from model_controller import *
from work_with_db import *
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
from my_utils import get_models_by_task
import os
import shutil
from datetime import timedelta
import subprocess
import psutil

import warnings
warnings.filterwarnings("ignore")

dataset_ready, dataset_name, history_eval, eval_result, show_text_area = False, False, False, False, False
st.session_state.show_train_logs = False

pid = os.getpid()

def disable(state):
    st.session_state["disabled"] = state
    st.session_state["first_time"] = False
if st.session_state.get("first_time", True):
    disable(False)

st.header('Cервис автоматической адаптации моделей трансформеров к задачам обработки естественного языка')

task = st.selectbox(
   "Выберите задачу",
   ("Text Classification", "Token Classification",  "Text Summarization",  "Masked Language Modeling",  "Question Answering", "Text Generation", ),
   index=None,
   placeholder="Выберите задачу...",
   disabled=st.session_state.get("disabled", True),
)
model_name = st.selectbox(
   "Выберите нейронную модель",
   get_models_by_task(task),#("roberta-base", "xlm-roberta-base"),
   index=None,
   placeholder="Выберите нейронную модель...",
   disabled=st.session_state.get("disabled", True),
)
quantization = st.selectbox(
    "Выберите метод квантования",
    ("don't use quantization",
     "use 8bit quantization",
     "use 4bit quantization"),
    help='Квантование существенно сокращает затраты на использование памяти графического процессора, позволяя, жертвуя точностью, загружать более крупные модели, которые обычно не помещаются в память.',
    disabled=st.session_state.get("disabled", True),
)
if quantization == "don't use quantization":
    quantization = False
    adapter_type = st.selectbox(
    "Выберите вид адаптера",
    ('Pfeiffer Bottleneck',
        'Houlsby Bottleneck',
        'Parallel Bottleneck',
        'Scaled Parallel Bottleneck',
        'Compacter',
        'Compacter++',
        'Prefix Tuning',
        'Flat Prefix Tuning',
        'LoRA',
        '(IA)^3',
        'MAM Adapter',
        'UniPELT',
        ),
    index=None,
    placeholder="Выберите вид адаптера...",
    disabled=st.session_state.get("disabled", True),
    )
else:
    adapter_type = st.selectbox("Выберите вид адаптера", ('LoRA',), placeholder="Выберите вид адаптера...", disabled=st.session_state.get("disabled", True))

lr_column, batch_col, epoch_col, length_col = st.columns([1, 1, 1, 1])
if adapter_type == '(IA)^3':
    default_lr = 8e-3
elif adapter_type == 'Prefix Tuning':
    default_lr = 9e-5
elif adapter_type == 'Flat Prefix Tuning':
    default_lr = 1e-3
elif adapter_type == 'UniPELT':
    default_lr = 5e-5
elif adapter_type == 'Compacter' or adapter_type == 'Compacter++':
    default_lr = 1e-3
else:
    default_lr = 1e-4
learning_rate = lr_column.select_slider('Select a learning rate', 
                                 options=[      2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
                                          1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
                                          1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3],
                                 value=default_lr,
                                 disabled=st.session_state.get("disabled", True))

batch_size = batch_col.select_slider('Select batch size', 
                                     options=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                                     value=8,
                                     disabled=st.session_state.get("disabled", True))

max_num_epoch = epoch_col.select_slider('Select max number of epochs (training can stop earlier)', 
                                     options=[1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100],
                                     value=10,
                                     disabled=st.session_state.get("disabled", True))

max_length = length_col.select_slider('Select max length of context / generation for the model', 
                                     options=[32, 64, 128, 256, 512, 768, 1024, 2048, 4096, 8192, 'max for model'],
                                     value=512,
                                     help = 'The larger max length, the more VRAM will be required for training',
                                     disabled=st.session_state.get("disabled", True))
if max_length == 'max for model':
    max_length = None

gradient_checkpointing_enable = st.checkbox("Включить контрольную точку градиента", help='Снижает затраты на память, жертвуя скоростью обучения', disabled=st.session_state.get("disabled", True))

#############
if task:
    dataset_names, idxs = get_datasets_names(task)
    idxs = [None]+idxs
    dataset_names = ['Другой датасет']+dataset_names
    idx = st.selectbox(
        label="Выберите датасет",
        options= range(len(dataset_names)),
        format_func=dataset_names.__getitem__,
        index=None,
        placeholder="Выберите датасет...",
        disabled=st.session_state.get("disabled", True)
    )
    if idx is not None:
        idx, dataset_name = idxs[idx], dataset_names[idx]
#############
    main_col = st.columns(1)[0]
    df_column, right_col= st.columns([1,1])
    if dataset_name == 'Другой датасет':
        with main_col:
            dataset_name = st.text_input("Введите название датасета")
        with right_col:
            if dataset_name:
                show_text_area = True
                dataset_ready = st.checkbox('saving this dataset configuration')
    elif dataset_name:
        with right_col:
            delete_ds = button('delete this dataset configuration', key="delete_ds", disabled=st.session_state.get("disabled", True))
            if delete_ds:
                _, sure_col= st.columns([1,5])
                with sure_col:
                    st.write('are you sure?')
                sure_col1, sure_col2= st.columns([1,1.5])
                with sure_col1:
                    sure_delete_ds = button('yes', key="sure")
                    if sure_delete_ds:
                        delete_dataset_name(idx)
                        del st.session_state.sure
                        del st.session_state.delete_ds
                        st.rerun()
                with sure_col2:
                    if st.button('cansel'):
                        del st.session_state.delete_ds
                        st.rerun()
        

#@st.cache_resource
def model_init(model_name, dataset_name, adapter_type, id2label, task, learning_rate, batch_size, max_num_epoch, quantization, gradient_checkpointing_enable, max_length):
    return modelka(model_name=model_name,ds_name=dataset_name, adapter_type=adapter_type, id2label=id2label, task=task, learning_rate=learning_rate, 
                   batch_size=batch_size, max_num_epoch=max_num_epoch, quantization=quantization, gradient_checkpointing_enable=gradient_checkpointing_enable,
                   max_length=max_length)

def restart_streamlit():
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    stop_threads = True
    subprocess.Popen('python start.py restart', shell=True)
    parent.kill()
    

def train_model(model):
    my_callbask = model.get_callback()
    max_steps = 1 if batch_size >=16 else 16//batch_size
    max_steps = model.get_dataset_len() * max_num_epoch // max_steps // batch_size
    mty = st.empty()
    
    def show_logs_table(stop):
        my_bar = st.progress(0, text='gradient steps')
        cancel_button = st.button("Cancel process", key='cancel_button', on_click=restart_streamlit)
        while True:
            if stop():
                break
            
            with mty.container():
                st.line_chart(pd.DataFrame(my_callbask.get_logs()))
            
            avg_step_time, current_steps, training_time = my_callbask.get_step_and_time()
            time_left = avg_step_time*(max_steps-current_steps)
            time_left = str(timedelta(seconds=time_left)).split(".")[0]; training_time = str(timedelta(seconds=training_time)).split(".")[0]
            my_bar.progress(current_steps/max_steps, text=f'{current_steps}/{max_steps} [{training_time} < {time_left}]')
            time.sleep(3)
            
        my_bar.empty()
                
    stop_threads = False
    thread_one = threading.Thread(target=show_logs_table, args =(lambda : stop_threads, ))
    add_script_run_ctx(thread_one)
    thread_one.start()
    eval_result, history_eval = model.train(my_callbask)
    stop_threads = True
    mty.empty()
    return eval_result, history_eval

def train_pipe(task, model_name, dataset_name, adapter_type, id2label={0: "👎", 1: "👍"}, learning_rate=1e-4, batch_size=1, max_num_epoch=50, quantization=False, gradient_checkpointing_enable=False, max_length=None):
    with st.spinner('Load model:'):
        model = model_init(model_name, dataset_name, adapter_type, id2label, task, learning_rate, batch_size, max_num_epoch, quantization, gradient_checkpointing_enable, max_length)
        model.model_load()
    with st.spinner('Load dataset:'):
        model.load_ds()
    with st.spinner('Training progress:'):
        eval_result, history_eval = train_model(model)
    adapter_name, adapter_save_path = model.save_model_adapter(task)
    row_id = post_finetuned_adapter(task, model_name, adapter_save_path, adapter_name, adapter_type, id2label, history_eval, learning_rate, quantization)
    model.clear_cuda_memory() #clear almost all memory
    del model
    return row_id, eval_result, history_eval, adapter_save_path

if task == 'Text Classification':
    if dataset_name:
        df = pd.DataFrame(get_datasets_classes(idx),columns=["classnumber","classname"])
        df.astype({'classnumber': int, 'classname': str})
        with df_column:
            df = st.data_editor(df, num_rows="dynamic", hide_index=True, disabled=st.session_state.get("disabled", True))
        class_df_ready = st.checkbox('classes ready', disabled=st.session_state.get("disabled", True))
        
        if dataset_ready and class_df_ready:
            with right_col:
                post_dataset_name(task, dataset_name, df.values.tolist())
                st.write('dataset saved ✅')
            
        if model_name and class_df_ready and adapter_type:
            id2label = {k:v for k, v in df.to_dict('split', index=False)['data']}
            train = st.button('train', on_click=disable, args=(True,), disabled=st.session_state.get("disabled", True))
            if train:
                row_id, eval_result, history_eval, adapter_save_path = train_pipe(task, model_name, dataset_name, adapter_type, id2label=id2label, learning_rate=learning_rate, batch_size=batch_size, max_num_epoch=max_num_epoch, quantization=quantization, gradient_checkpointing_enable=gradient_checkpointing_enable, max_length=max_length)
                
elif task == 'Text Generation':
    if dataset_name:
        with df_column:
            if show_text_area:
                txt = st.text_area("Необязательно! Введите описание датасета...", disabled=st.session_state.get("disabled", True))
            else:
                txt = get_datasets_text(idx)
                st.write(txt if txt else 'Description is empty...')#, disabled=st.session_state.get("disabled", True))
        class_df_ready = st.checkbox('all is ready', disabled=st.session_state.get("disabled", True))
        
        if dataset_ready and class_df_ready:
            with right_col:
                post_dataset_name(task, dataset_name, txt)
                st.write('dataset saved ✅')
            
        if model_name and class_df_ready and adapter_type:
            train = st.button('train', on_click=disable, args=(True,), disabled=st.session_state.get("disabled", True))
            if train:
                row_id, eval_result, history_eval, adapter_save_path = train_pipe(task, model_name, dataset_name, adapter_type, learning_rate=learning_rate, batch_size=batch_size, max_num_epoch=max_num_epoch, quantization=quantization, gradient_checkpointing_enable=gradient_checkpointing_enable, max_length=max_length) 
                

def remove_and_rerun(row_id, adapter_save_path):
    remove_adapter(row_id)
    shutil.rmtree(adapter_save_path)
    disable(False)
    restart_streamlit()

def just_rerun():
    disable(False)
    restart_streamlit()

if history_eval and eval_result:
    line_chart_col = st.columns(1)[0]
    with line_chart_col:
        tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
        tab1.subheader("Training progress line:")
        tab1.line_chart(pd.DataFrame(history_eval))
        tab2.subheader("Evaluation final result:")
        tab2.write(eval_result)
        
        st.write('Adapter saved. You can remove it if you are not satisfied with the evaluation results')
        rerun_col, delete_adapter_col = st.columns([1,1])

        rerun_col.button('complete', on_click=just_rerun)
        delete_adapter_col.button('remove adapter', on_click=remove_and_rerun, args=(row_id, adapter_save_path),)
                