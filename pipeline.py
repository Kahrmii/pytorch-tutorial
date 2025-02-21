from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/switch-c-2048")
model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-c-2048", device_map="auto", offload_folder="C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial")

input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))