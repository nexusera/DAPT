# encoding:utf-8

import requests
import base64
import json
import urllib
from typing import Literal

def ocr(base64_img,mode:Literal["accurate","general_v6"]="accurate"):
    """
    :param base64_img: 图片的base64编码
    :return: 检测到的文字信息
    """
    # 无论传入什么 mode，都强制使用 V6 逻辑，因为 URL 只支持 V6
    # 且为了兼容性，需要将 V6 的返回结果转换为旧版通用 OCR 的格式

    request_url = "https://prod-t4-baidu.aistarfish.net/GeneralClassifyService/classify"

    # V6 接口参数构造
    request_str_params = {
        # --- 核心参数 ---
        "object_type": "general_v5",
        "type": "st_ocrapi_all", 
        "languagetype": "CHN_ENG",
        
        # --- 表格参数 (按需开启) ---
        # "line_probability": "true",        # =true则返回行置信度
        "imgDirection": "setImgDirFlag",     # 选填，开启方向判断
        # "RecOnly": "setRecOnlyFlag",       # 可选，跳过检测，直接当作单行识别
        # "LineDirection": "setLineDirFlag", # 选填，文本行级别的方向矫正
        # "auto_detect_langtype": "true",    # 自动检测语种 (需同时设置 languagetype=CHN_ENG)
        "eng_granularity": "word",           # 可选 {word, letter}
        "disp_line_poly": "true",            # =true 返回行的多点坐标位置信息
        # "disp_paragraph_poly": "true",     # 可选，是否返回段落结果
        # "file_type": "pdf",                # 可选，仅当文件是pdf时填
        # "file_page_number": "0",           # 单次只能返回一页内容，0表示第1页

        # --- 其他 ---
        "disp_chars": "true",                # (保持开启以获取单字位置)
    }

    try:
        # 1. 构造请求 Payload
        params_str = urllib.parse.urlencode(request_str_params)
        full_data_str = params_str + "&image=" + base64_img
        start_encoded = base64.b64encode(full_data_str.encode("utf-8")).decode("utf-8")
        
        req_payload = {
            "provider": "default",
            "data": start_encoded
        }
        headers = {'Content-Type': 'application/json'}

        # 2. 发送请求
        response = requests.post(request_url, data=json.dumps(req_payload), headers=headers)
        
        # 3. 解析响应
        if response.ok:
            try:
                res_json = response.json()
            except json.JSONDecodeError:
                print(f"OCR Response Not JSON. Status: {response.status_code}, Text: {response.text}")
                return {"err_no": -1, "err_msg": "JSONDecodeError", "text": response.text}
            
            # 4. 解码 RPC 层结果 (Base64 decode 'result')
            business_res = res_json
            if "result" in res_json and isinstance(res_json["result"], str):
                try:
                    # 尝试解码 result 字段
                    decoded_str = base64.b64decode(res_json["result"]).decode("utf-8")
                    business_res = json.loads(decoded_str)
                except Exception as e:
                    print(f"OCR Result Decode Error: {e}")
                    # 如果解码失败，可能本身就不是 base64，继续尝试直接使用 res_json
            
            # 5. 格式适配 (V6 -> Legacy/Accurate Format)
            # 如果是 V6 格式 (含有 ret)，则转换为 words_result 格式以便兼容 downstream (如 ocr_split.py)
            if "ret" in business_res:
                legacy_res = {
                    "words_result": [],
                    "words_result_num": 0,
                    "paragraphs_result": [],
                    "log_id": business_res.get("logid") or business_res.get("log_id")
                }
                
                # 转换 lines (ret -> words_result)
                for item in business_res["ret"]:
                    legacy_item = {
                        "words": item.get("word", ""),
                        "location": item.get("rect", {}) # rect 结构 {left, top, width, height} 与 location 兼容
                    }
                    if "probability" in item:
                         legacy_item["probability"] = item["probability"]
                    
                    # 转换 chars (charset -> chars)
                    if "charset" in item:
                        legacy_chars = []
                        for char_item in item["charset"]:
                             l_char = {
                                 "char": char_item.get("word", ""),
                                 "location": char_item.get("rect", {})
                             }
                             if "prob" in char_item:
                                 l_char["probability"] = char_item["prob"]
                             legacy_chars.append(l_char)
                        legacy_item["chars"] = legacy_chars

                    legacy_res["words_result"].append(legacy_item)
                
                legacy_res["words_result_num"] = len(legacy_res["words_result"])

                # 转换 paragraphs (paragraphs -> paragraphs_result)
                if "paragraphs" in business_res:
                     for para in business_res["paragraphs"]:
                         legacy_para = {
                             "words_result_idx": para.get("para_idx", [])
                         }
                         legacy_res["paragraphs_result"].append(legacy_para)
                         
                return legacy_res
            
            # 如果不是 V6 格式 (或者是错误信息)，直接返回原样
            return business_res
            
        else:
            print(f"OCR Request Failed. Status: {response.status_code}, Text: {response.text}")
            return {"err_no": response.status_code, "err_msg": "Request Failed", "text": response.text}

    except Exception as e:
        print(f"OCR Execution Error: {e}")
        return {"err_no": -1, "err_msg": str(e)}

            



def extract_all_paragraph_objects(words_result):
    """
    提取所有段落的对象，保持数据原结构。

    :param words_result: 包含所有文字信息和段落信息的列表
    :return: 包含所有段落对象的列表
    """
    paragraphs_result = words_result.get("paragraphs_result", [])
    all_paragraph_objects = []

    for paragraph in paragraphs_result:
        words_indices = paragraph["words_result_idx"]
        paragraph_objects = [words_result["words_result"][idx] for idx in words_indices]
        all_paragraph_objects.append(paragraph_objects)

    return all_paragraph_objects


if __name__ == "__main__":
    img_path = ""
    res = ocr(img_path)
    # print("---------------------------")

    # paragraph = extract_all_paragraph_objects(res)
    # print(paragraph)
