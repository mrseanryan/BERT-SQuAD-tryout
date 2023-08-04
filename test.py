import json
from bert import QA

model = QA('model')

doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the state's law-making body for matters coming under state responsibility. The Victorian Constitution can be amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian people in a referendum, depending on the provision."

def send_prompt(q):
    print(f"""
        doc:
        {doc}

        question: {q}
    """)

    answer = model.predict(doc,q)

    print(f"""
        Answer: {answer['answer']}

        Keys: {answer.keys()}

        Full answer: {json.dumps(answer)}
    """)

q = 'When did Victoria enact its constitution?'

# 1975
# dict_keys(['answer', 'start', 'end', 'confidence', 'document']))

send_prompt(q)

send_prompt("Create a short summary of the doc.")
