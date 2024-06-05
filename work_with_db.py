import sqlite3
import json

def get_tasks():
    with sqlite3.connect('tasks_models_adapters.db') as db:
        tasks = db.execute('SELECT DISTINCT task FROM tasks_models_adapters').fetchall()
    return [i[0] for i in tasks]

def get_models_by_task(task):
    with sqlite3.connect('tasks_models_adapters.db') as db:
        models = db.execute('SELECT DISTINCT model FROM tasks_models_adapters WHERE task = ?', (task,)).fetchall()
    return [i[0] for i in models]

def get_adapters_by_task_and_model(task, model):
    with sqlite3.connect('tasks_models_adapters.db') as db:
        rows = db.execute('SELECT DISTINCT adapter_name, adapter_type, creation_date, id, bit4, bit8 FROM tasks_models_adapters WHERE task = ? AND model = ? ORDER BY creation_date ASC', (task,model,)).fetchall()
    quant = {row:('4bit',) if row[4] else ('8bit',) if row[5] else () for row in rows}
    return [' | '.join(row[0:2]+quant[row]+row[2:3]) for row in rows], [row[3] for row in rows]

def get_row(idx):
    with sqlite3.connect('tasks_models_adapters.db') as db:
        row = db.execute('SELECT adapter_path, tags, evaluation, bit4, bit8 FROM tasks_models_adapters WHERE id = ?', (idx,)).fetchall()[0]
    return row[0], json.loads(row[1]), json.loads(row[2]), '4bit' if row[3] else '8bit' if row[4] else False


def get_datasets_classes(idx):
    if idx is None:
        return [[0,None], [1,None]]
    with sqlite3.connect('tasks_models_adapters.db') as db:
        rows = db.execute("SELECT cols FROM datasets WHERE id = ?", (idx,)).fetchall()[0]
    if rows:
        return [json.loads(row) for row in rows][0]
    else:
        return [[0,None], [1,None]]
    
def get_datasets_text(idx):
    if idx is None:
        return None
    with sqlite3.connect('tasks_models_adapters.db') as db:
        rows = db.execute("SELECT cols FROM datasets WHERE id = ?", (idx,)).fetchall()[0]
    if rows:
        return rows[0]
    else:
        return None
    
def get_datasets_names(task):
    with sqlite3.connect('tasks_models_adapters.db') as db:
        rows = db.execute("SELECT dataset_name, id FROM datasets WHERE task = ?", (task,)).fetchall()
    return [row[0] for row in rows], [row[1] for row in rows]

def post_dataset_name(task, ds_name, tags):
    with sqlite3.connect('tasks_models_adapters.db') as db:
        if isinstance(tags, list):
            for i, el in enumerate(tags):
                tags[i][0] = int(tags[i][0])#[[0.0,'👎'], [1.1,'👍']] -> [[0,'👎'], [1,'👍']]
            tags = json.dumps(tags)
        db.execute("INSERT OR IGNORE INTO datasets (dataset_name, cols, task) VALUES(?, ?, ?)", (ds_name, tags, task))
        db.commit()
        
def delete_dataset_name(idx):
    with sqlite3.connect('tasks_models_adapters.db') as db:
        db.execute("DELETE FROM datasets WHERE id = ?", (idx,))
        
def post_finetuned_adapter(task, model_name, adapter_save_path, adapter_name, adapter_type, id2label, history_eval, learning_rate, quantization):
    bit4, bit8 = 0, 0
    if quantization == 'use 4bit quantization':
        bit4 = 1
    elif quantization == 'use 8bit quantization':
        bit8 = 1
        
    with sqlite3.connect('tasks_models_adapters.db') as db:
        history_eval = json.dumps(history_eval)
        id2label = json.dumps(id2label)
        db.execute("INSERT OR IGNORE INTO tasks_models_adapters (task, model, adapter_path, adapter_name, adapter_type, tags, evaluation, learning_rate, bit4, bit8) VALUES(?,?,?,?,?,?,?,?,?,?)", 
                   (task, model_name, adapter_save_path, adapter_name, adapter_type, id2label, history_eval, learning_rate, bit4, bit8))
        db.commit()
        last_id = db.execute('SELECT id FROM tasks_models_adapters WHERE (task, model, adapter_path, adapter_name, adapter_type, tags, evaluation, learning_rate, bit4, bit8) = (?,?,?,?,?,?,?,?,?,?)',
                   (task, model_name, adapter_save_path, adapter_name, adapter_type, id2label, history_eval, learning_rate, bit4, bit8)).fetchall()[0][0]
    return last_id
    
def remove_adapter(row_id):
    with sqlite3.connect('tasks_models_adapters.db') as db:
        db.execute("DELETE FROM tasks_models_adapters WHERE id = ?", (row_id,))
        db.commit()