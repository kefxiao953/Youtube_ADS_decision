#==================================================
#Langchain是大模型LLM的开发框架，最近两年非常火，因此摸索学习是有必要的
#但是Langchain本身依然在快速的变化中，在你看到的时候，本段代码很可能已经与当前langchain版本不一致，无法直接运行。
#因此此代码仅作为参考。
#要想让本段代码成功运行，需要你结合Langchain官网文档进行调试。
#需要注意的是，langchain+LLM在本项目中要实现的目标，是进行评论文本的自动生成。
#即给我们投流的视频进行自动评论，以增加人气。
#==================================================

#要调通本段代码，需要你先去Huggingface上或openai上新建一个账户，然后下载相应的文本生成模型
from langchain import HuggingFaceHub,LLMChain
import os

#配置 api_token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xxx"#这里写上你的huggingface api_token
#我这里用到的是flan_ul2和flan_t5模型
flan_ul2 = HuggingFaceHub(
    repo_id="google/flan-ul2", 
    model_kwargs={"temperature":0.1,
                  "max_new_tokens":256})

flan_t5 = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"temperature":0 }
)

#调库
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
#使用langchain的机制来实现，先要有一个memory
memory = ConversationBufferMemory()

#创建coversation，这是lanchain的写法
conversation = ConversationChain(
    llm=flan_ul2, 
    verbose=True, 
    memory=memory
)

#使用conversation，尝试进行prediction，让LLM给你回答问题
conversation.predict(input="Hi there! I am Sam")

conversation.predict(input="How are you today?")

conversation.predict(input=" Can you help me with some customer support?")

conversation.predict(input="My TV is broken. can you help fix it?")

#让LLM生成评论，改变提示词，让他生成即可
conversation.predict(input="Generate some comments for me")

#制作一个自动问答机器人
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

#测一下
test_input = "Now Good Morning Ms Rogers"
# tokenizer([test_input])
tokenizer.tokenize(test_input)

#用chat_to_llm函数实现自动解析人类语言，然后交由LLM进行回答
def chat_to_llm(chat_llm):
    conversation_total_tokens = 0
    new_conversation = ConversationChain(llm=chat_llm, 
                                     verbose=False, 
                                     memory=ConversationBufferMemory()
                                     )

    while True:
        message = input("Human: ")
        if message=='exit':
            print(f"{conversation_total_tokens} tokens used in total in this conversation")
            break
        if message:
            formatted_prompt = conversation.prompt.format(input=message,history=new_conversation.memory.buffer) # 导入对话历史
            num_tokens = len(tokenizer.tokenize(formatted_prompt))
            conversation_total_tokens += num_tokens
            print(f'tokens sent {num_tokens}')
            response = new_conversation.predict(input=message)
            response_num_tokens = len(tokenizer.tokenize(response))
            conversation_total_tokens += response_num_tokens
            print(f"LLM: {response}")

#运行该函数
chat_to_llm(flan_ul2)
