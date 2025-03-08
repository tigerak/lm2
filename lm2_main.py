import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from time import time

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# huggingface
from transformers import (Trainer, TrainingArguments,
                          AutoTokenizer, AutoModelForCausalLM, 
                          LlamaConfig,
                          BitsAndBytesConfig)

from config import *
from function.lm2.memory_module import LM2MemoryModule
from function.lm2.lm2causal import LM2ForCausalLM
from function.lm2.lm2_dataset import LM2_Dataset, collate_fn


if __name__=="__main__":

    use_bf16 = True

    lm2_config= LlamaConfig.from_pretrained(LLAMA3_2_KO_3B)
    # print(lm2_config)
    lm2_config.hidden_size = 256
    lm2_config.num_attention_heads = 4
    lm2_config.num_hidden_layers = 16
    lm2_config.num_key_value_heads = 4
    
    lm2_config.num_slots = 8
    print(lm2_config)

    d_model = lm2_config.hidden_size
    num_slots = lm2_config.num_slots
    memory_module = LM2MemoryModule(d_model=d_model,
                                    num_slots=num_slots)
    

    model = LM2ForCausalLM(config=lm2_config,
                           memory_module=memory_module).cuda()
    if use_bf16:
        model = model.to(dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained('beomi/OPEN-SOLAR-KO-10.7B')

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    ### 기존 학습 모델 불러오기 ###
    leaning_model_path = r'/home/lab_ai/Project/output/last_model.pt'
    start_epoch = 0
    new_mem = None
    if os.path.exists(leaning_model_path):
        print(f"기존 학습 모델 로드 시작: {leaning_model_path}")

        checkpoint = torch.load(leaning_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if use_bf16:
            model = model.to(dtype=torch.bfloat16)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] 
        start_step = checkpoint["step"]
        new_mem = checkpoint["memory_states"]
        model.to("cuda")
        print(f"✅ 모델 로드 완료! {start_epoch}epoch 부터 학습을 재개합니다.")
    else:
        print(f"🚨 저장된 모델이 없습니다. 새 모델을 학습합니다.")
        start_step = 0

    print("모델 준비 !!!!!!")
    
    
    
    # 데이터 생성성
    dataset = LM2_Dataset(json_path=YTN_DATA, 
                            tokenizer=tokenizer)
    
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=collate_fn)
    
    # training_args = TrainingArguments(
    #                         output_dir="./output",  # 모델 저장 경로
    #                         # evaluation_strategy="epoch",  # 매 epoch마다 검증 실행
    #                         save_strategy="epoch",  # 매 epoch마다 모델 저장
    #                         per_device_train_batch_size=1,  # GPU당 배치 크기
    #                         # per_device_eval_batch_size=1,
    #                         gradient_accumulation_steps=64,  # 메모리 절약을 위해 gradient 누적
    #                         learning_rate=5e-5,  # 학습률
    #                         weight_decay=0.01,  # 가중치 감쇠 (AdamW)
    #                         num_train_epochs=3,  # 학습할 epoch 수
    #                         save_total_limit=2,  # 저장할 모델 체크포인트 개수 제한
    #                         logging_dir="./output/logs",  # 로그 저장 경로
    #                         logging_steps=1,  # 10 스텝마다 로그 출력
    #                         # fp16=True,  # mixed precision (속도 향상)
    #                         bf16=True,
    #                         push_to_hub=False,  # 학습 완료 후 허브에 업로드 여부
    #                     )
    
    # trainer = Trainer(model=model,
    #                     args=training_args,
    #                     train_dataset=dataset)
    
    # trainer.train()

    # 결과 저장을 위한 리스트
    generation_results = []
    
    ### 학습 시작 ###
    total_epoch = 3
    accum_steps = 8
    step_epoch = len(dataloader)
    total_steps = step_epoch/accum_steps
    print(f"총 데이터 수: {dataset.__len__()} | 실제 배치: {step_epoch} | 배치 합산: {accum_steps}")
    for epoch in range(start_epoch, total_epoch):
        model.train()
        
        current_step = 0
        for step, batch in enumerate(dataloader):
            if start_step:
                current_step = (step + start_step)
            else:
                current_step = step
            display_step = current_step // accum_steps
            start_time = time()

            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            
            logit, loss, new_mem = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels,
                                        memory_states=None)
            # new_mem = new_mem.detach() # Autograd 추적 차단
            loss = loss / accum_steps
            if loss < 1e-6:
                loss = torch.tensor(1e-6, requires_grad=True).to(loss.device)
            loss.backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient 폭발 방지
                optimizer.step()
                optimizer.zero_grad()
                
                end_time = time()

                for i in range(5):
                    i += 30
                    input_word = tokenizer.decode(input_ids[0, i].item(), skip_special_tokens=False)
                    target_word = tokenizer.decode(labels[0, i+1].item(), skip_special_tokens=False) if labels[0, i].item() != -100 else "[IGNORE]"
        
                    predicted_token_id = logit[0, i].argmax(dim=-1).item()
                    predicted_word  = tokenizer.decode(predicted_token_id, skip_special_tokens=False)
                    
                    print(f"🔹 입력: '{input_word}' | 정답: '{target_word}' | 예측: '{predicted_word }'")

                print(f"[Epoch:{epoch} Step:{display_step}/{total_steps}] loss={loss.item():.4f} time:{end_time-start_time:.2f}")

            # if step == 3:
            #     break

            # 🔹 모델 저장 (각 step마다 저장)
            model_save_path = os.path.join(save_dir, f"last_model.pt")
            torch.save({
                "epoch": epoch,
                "step": current_step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "memory_states": new_mem, ####
            }, model_save_path)
            
            print(f"✅ 모델 저장 완료: {model_save_path}")

            if total_steps - display_step < 1:
                break

        leftover = step_epoch % accum_steps
        if leftover != 0:
            optimizer.step()
            optimizer.zero_grad()

        model.eval()

        # test_text = "한국전력이 잇따른 전기요금 인상과 국제 연료비 인하 효과 덕분에 지난해 4년 만에 흑자 전환에 성공했습니다. 한전은 연결 기준 지난해 영업이익을 8조 3천억 원으로 잠정 집계했습니다. 이에 따라 2021년 이후 한전의 누적 영업적자는 34조 7천억 원으로 줄었습니다."
        
        test_title = "### TITLE: 한전, 지난해 영업익 8조 3천억...4년 만에 흑자 전환\n###ARTICLE: "
        encoding = tokenizer(text=test_title, 
                            truncation=True,
                            max_length=1024,
                            add_special_tokens=True,
                            return_tensors="pt").to("cuda") 
        input_ids = encoding["input_ids"]

        correct_predictions = 0
        total_predictions = 0

        # with torch.no_grad():
        #     output = model.generate(input_ids, max_new_tokens=100, do_sample=False)

        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # print(f"\n🔹 [Epoch {epoch+1}] 모델 평가 결과:")
        # print(f"입력: {test_text}")
        # print(f"출력: {generated_text}")

        generated = input_ids.clone()
        memory_states = new_mem #########
        for i in range(100):
            with torch.no_grad():
                out_logits, _, memory_states = model(
                    input_ids=generated,
                    attention_mask=torch.ones_like(generated).to("cuda"), # for문으로는 필요요
                    memory_states=memory_states
                )
            next_token = out_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

        generated_text = "generated:", tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)

        print(f"\n🔹 [Epoch {epoch+1}] 모델 평가 결과:")
        print(f"입력: {test_title}")
        print(f"출력: {generated_text}")

        # 🔹 결과 저장
        generation_results.append({
            "epoch": epoch + 1,
            "input": test_title,
            "generated": generated_text
        })

        # 🔹 모델 저장 (각 epoch마다 저장)
        model_save_path = os.path.join(save_dir, f"last_model.pt")
        # model_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "memory_states": new_mem, ######################
        }, model_save_path)
        
        print(f"✅ 모델 저장 완료: {model_save_path}")

        # 🔹 최종 결과를 JSON 파일로 저장
        results_path = os.path.join(save_dir, "generation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(generation_results, f, ensure_ascii=False, indent=4)
            # tokenizer = AutoTokenizer.from_pretrained(LLAMA3_2_KO_3B)
    