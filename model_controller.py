from datasets import load_dataset
from adapters import init, AutoAdapterModel, AdapterTrainer
from adapters import BnConfig, SeqBnConfig, DoubleSeqBnConfig, PrefixTuningConfig, LoRAConfig, IA3Config, PromptTuningConfig, MAMConfig, UniPELTConfig, ParBnConfig, CompacterConfig, CompacterPlusPlusConfig
import numpy as np
import os
import gc
from torch import float16, bfloat16, float32
from torch.cuda import empty_cache, is_available
from torch.nn import Sequential
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import pipeline
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
import evaluate
from peft import PeftModel, PeftConfig
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification
from peft import get_peft_model, prepare_model_for_kbit_training
from peft import LoraConfig as QLoraConfigPEFT
from datetime import datetime

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

evaluation_strategy='epoch'
early_stopping_patience = 5
save_total_limit = 1
accuracy = evaluate.load("accuracy")

class CustomCallback(TrainerCallback):
    def __init__(self):
        self.my_own_logs = []
        self.avg_step_time, self.current_steps, self.training_time = 0,0,0
    
    def on_evaluate(self, args, state, control, **kwargs):
        try:
            if len(state.log_history) == 3:
                self.my_own_logs[0].update({'loss':state.log_history[0]['loss']})
            logs = state.log_history[-2:]
            logs = {'loss':logs[0]['loss'], 'eval_loss':logs[1]['eval_loss']}
            #logs = {**logs[0], **logs[1]}
            self.my_own_logs.append(logs)
        except: pass

    def on_step_end(self, args, state, control, **kwargs):
        self.current_steps += 1
        self.training_time = (datetime.now() - self.start_time).total_seconds()
        self.avg_step_time = self.training_time / self.current_steps
        
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self.current_steps = 0
        
    def get_step_and_time(self):
        return self.avg_step_time, self.current_steps, self.training_time
    
    def add_val(self, val):
        self.my_own_logs.append(val)

    def get_logs(self):
        return self.my_own_logs

class modelka:
    def __init__(self,
                model_name="roberta-base",
                ds_name="rotten_tomatoes",
                learning_rate=1e-4,
                adapter_name=None,
                adapter_type='Pfeiffer Bottleneck',
                id2label={0: "👎", 1: "👍"},
                task='Text Classification',
                batch_size=1,
                max_num_epoch=50,
                quantization=False,
                gradient_checkpointing_enable=False,
                max_length=None
                ):
        self.task = task
        self.quantization = quantization
        self.gradient_checkpointing_enable = gradient_checkpointing_enable
        self.model_name = model_name
        self.id2label = id2label
        self.adapter_type = adapter_type
        self.max_length = max_length
        self.label2id = {v:k for k,v in id2label.items()}
        print(ds_name)
        if ',' in ds_name:
            ds_name, ds_task_name = ds_name.split(',')
            self.ds_name = ds_name.strip()
            self.ds_task_name = ds_task_name.strip()
        else:
            self.ds_name = ds_name
            self.ds_task_name = None
        self.adapter_name = adapter_name if adapter_name else ds_name.replace('/', '_')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        
        self.training_args = TrainingArguments(
            fp16=True if not quantization else False,
            bf16=True if quantization else False,
            learning_rate=learning_rate,#1e-4
            num_train_epochs=max_num_epoch,
            auto_find_batch_size=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1 if batch_size >=16 else 16//batch_size,
            logging_strategy="epoch", # "steps",
            logging_first_step=True,
            #logging_steps=log_steps,
            save_strategy=evaluation_strategy,
            #save_steps=log_steps,
            evaluation_strategy=evaluation_strategy,
            output_dir="./training_output",
            overwrite_output_dir=True,
            remove_unused_columns=True,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            #group_by_length=True,
            lr_scheduler_type="constant",
            optim="paged_adamw_8bit",
            label_names=["input_ids"] if self.task == 'Text Generation' else ['labels'],
            warmup_ratio=0.03,
            #torch_compile=True,
            prediction_loss_only=True,
        )
    
    def load_ds(self):
        print(self.ds_name, self.ds_task_name) if self.ds_task_name else print(self.ds_name)
        dataset = load_dataset(self.ds_name, self.ds_task_name) if self.ds_task_name else load_dataset(self.ds_name)
        
        if self.task=='Text Classification':
            def encode_batch(batch):
                return self.tokenizer(batch["text"], truncation=True, padding=True, max_length=self.max_length)
            dataset = dataset.map(encode_batch, batched=True)#"rotten_tomatoes"
            dataset = dataset.rename_column("label", "labels")
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            #dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            
        elif self.task=='Text Generation':
            # ############
            # dataset['train'] = dataset['train'].select(range(16))
            # dataset['test'] = dataset['test'].select(range(16))
            # #dataset['validation'] = dataset['validation'].select(range(100))
            # ############
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            def encode_batch(batch):
                return self.tokenizer(batch["text"], truncation=True, padding='max_length', max_length=self.max_length)
            dataset = dataset.map(encode_batch, remove_columns=dataset["train"].column_names, batched=True)#, num_proc=num_proc)
            
            self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        self.dataset_len = len(dataset['train'])
        self.dataset = dataset
    
    def get_dataset_len(self):
        return self.dataset_len
        
    def model_load(self):
        if self.adapter_type == 'Pfeiffer Bottleneck':
            adapter_config = SeqBnConfig()
        # elif self.adapter_type == '':
        #     adapter_config = SeqBnInvConfig()
        elif self.adapter_type == 'Houlsby Bottleneck':
            adapter_config = DoubleSeqBnConfig()
        # elif self.adapter_type == '':
        #     adapter_config = DoubleSeqBnInvConfig()
        elif self.adapter_type == 'Parallel Bottleneck':
            adapter_config = ParBnConfig()
        elif self.adapter_type == 'Scaled Parallel Bottleneck':
            adapter_config = ParBnConfig(scaling="learned")
        elif self.adapter_type == 'Compacter':
            adapter_config = CompacterConfig()
        elif self.adapter_type == 'Compacter++':
            adapter_config = CompacterPlusPlusConfig()
        elif self.adapter_type == 'Prefix Tuning':
            adapter_config = PrefixTuningConfig()
        elif self.adapter_type == 'Flat Prefix Tuning':
            adapter_config = PrefixTuningConfig(flat=True)
        elif self.adapter_type == 'LoRA':
            if self.quantization:
                task_type="CAUSAL_LM" if self.task == 'Text Generation' else "SEQ_CLS" if self.task == 'Text Classification' else None
                adapter_config = QLoraConfigPEFT(r=16,lora_alpha=32,lora_dropout=0.01,bias="none",task_type=task_type)
            else:
                adapter_config = LoRAConfig(r=16, alpha=32, dropout=0.01)
        elif self.adapter_type == '(IA)^3':
            adapter_config = IA3Config()
        # elif self.adapter_type == 'Prompt Tuning':
        #     adapter_config = PromptTuningConfig()
        # elif self.adapter_type == '':
        #     adapter_config = ConfigUnion()
        elif self.adapter_type == 'MAM Adapter':
            adapter_config = MAMConfig()
        elif self.adapter_type == 'UniPELT':
            adapter_config = UniPELTConfig()
        # elif self.adapter_type == '':
        #     adapter_config = AdapterFusionConfig()
        # elif self.adapter_type == '':
        #     adapter_config = StaticAdapterFusionConfig()
        # elif self.adapter_type == '':
        #     adapter_config = DynamicAdapterFusionConfig()
        
        if self.quantization == 'use 4bit quantization':
            bnb_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16,
                llm_int8_skip_modules=["classifier", "pre_classifier"]
            )
            torch_dtype_model = bfloat16
        elif self.quantization == 'use 8bit quantization':
            bnb_config=BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["classifier", "pre_classifier"])
            torch_dtype_model = bfloat16
        else:
            bnb_config = None
            torch_dtype_model = 'auto'
        
        if self.task=='Text Generation':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, quantization_config=bnb_config, torch_dtype=bfloat16)
            if not self.quantization:
                init(self.model)
            
        elif self.task=='Text Classification':
            if not self.quantization:
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoAdapterModel.from_pretrained(self.model_name, trust_remote_code=True, config=config)
                self.model.add_classification_head(self.adapter_name, num_labels=len(self.id2label), id2label=self.id2label)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, trust_remote_code=True, quantization_config=bnb_config, torch_dtype=torch_dtype_model, num_labels=len(self.id2label), id2label=self.id2label, label2id=self.label2id)
        
        self.model.config.use_cache = False
        
        if self.quantization:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, adapter_config)
            print(self.model.print_trainable_parameters())
        else:
            self.model.add_adapter(self.adapter_name, config=adapter_config)
            self.model.train_adapter(self.adapter_name)
            self.model.adapter_to(self.adapter_name, device="cuda")
            print(self.model.adapter_summary())
        
        if self.gradient_checkpointing_enable:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

    def get_callback(self):
        return CustomCallback()

    def train(self, my_callbask):
        def compute_accuracy(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        kwargs = {'model':self.model,
                  'args':self.training_args,
                  'train_dataset':self.dataset["train"],
                  'eval_dataset':self.dataset["test"],
                  'compute_metrics':compute_accuracy if self.task != 'Text Generation' else None,
                  'data_collator':self.data_collator,
                  'callbacks':[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience), my_callbask],
                  }
        if self.quantization:
            trainer = Trainer(**kwargs)
        else:
            trainer = AdapterTrainer(**kwargs)
        
        my_callbask.add_val({'eval_loss':trainer.evaluate()['eval_loss']})
        
        if not self.quantization:
            trainer.model.to('cuda')
            trainer.model.adapter_to(self.adapter_name, device="cuda")
        
        trainer.train()
        if not self.quantization:
            trainer.model.to('cuda')
            trainer.model.adapter_to(self.adapter_name, device="cuda")
        eval_result = trainer.evaluate()
        del trainer
        return eval_result, my_callbask.get_logs()

    def clear_cuda_memory(self):
        del self.model
        gc.collect()
        empty_cache()
    
    def save_model_adapter(self, task, path=f"./trained_adapters"):
        time_now = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        path = f'{path}/{task}/{self.model_name}/{self.adapter_name}/{self.adapter_type}/{time_now}'
        os.makedirs(path, exist_ok=True)
        if not self.quantization:
            self.model.save_adapter(path, self.adapter_name, with_head=True)
        else:
            self.model.save_pretrained(path)
        print(f'adapter was saved in path, {path}')     
        return self.adapter_name, path
            
class modelka_inference:
    def __init__(self, model_name, adapter_path, task='Text Classification', quantization=False, tags=None):
        if tags:
            tags = {int(k):v for k,v in tags.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.quantization = quantization
        
        if task=='Text Classification':
            if quantization:
                model = AutoPeftModelForSequenceClassification.from_pretrained(adapter_path, id2label=tags, trust_remote_code=True, torch_dtype=bfloat16, load_in_4bit=quantization=='use 4bit quantization', load_in_8bit=quantization=='use 8bit quantization', device_map="auto")
            else:
                model = AutoAdapterModel.from_pretrained(model_name, trust_remote_code=True)
                model.load_adapter(adapter_path, with_head=True, set_active=True)
            print(model.config.id2label)
            #model.config.label2id = {v:k for k,v in model.config.id2label.items()}
            
        elif task=='Text Generation':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if quantization:
                model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, trust_remote_code=True, torch_dtype=bfloat16, load_in_4bit=quantization=='4bit', load_in_8bit=quantization=='8bit', device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=bfloat16)
                init(model)
                model.load_adapter(adapter_path, load_as='gen', set_active=True)
                if is_available():
                    model.adapter_to("gen", device="cuda", dtype=bfloat16)

        if quantization:
            model = model.merge_and_unload()
        self.model = model
        
    def return_text_classifier(self):
        return pipeline("text-classification",model=self.model, tokenizer=self.tokenizer)#, torch_dtype=bfloat16 if self.quantization else 'auto')
    
    def return_text_generation(self):
        return pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, torch_dtype=bfloat16)