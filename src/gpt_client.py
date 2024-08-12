from openai import OpenAI
from typing import List, Dict

model = "gpt-3.5-turbo"
messages = [
    {"role": "system", "content": "You are an assistant that modifies a scene description. "
                                  "You will be given a scene description and a foreground object. "
                                  "The scene description will be provided to you after this text: 'Scene description: ' "
                                  "The foreground object will be provided to you after this text: 'Foreground object: ' "
                                  "You will remove the foreground object from the scene description. "
                                  "After the modification, only the background information must be present in the description. "
                                  "The only reply you give will be the new scene description where the foreground object is not present. "
                                  "You will strictly not give any other response other than the new scene description."},
    {"role": "user", "content": "Scene description: Bear wanders in a forest. Foreground object: bear"},
    {"role": "assistant", "content": "A forest"},
    {"role": "user", "content": "Scene description: A person sits on a chair inside a cafeteria. Foreground object: person"},
    {"role": "assistant", "content": "A cafeteria with chairs inside"},
    {"role": "user", "content": "Scene description: People playing basketball on a court. Foreground object: basketball"},
    {"role": "assistant", "content": "A basketball court with people in it"}
]
user_prompt = {"role": "user", "content": ""}
user_message = "Scene description: {scene_desc}. Foreground object: {fg_object}."


class GPTClient(OpenAI):

    def __init__(self):
        super().__init__()

    def remove_foreground(self, scene_desc: str, fg_object: str) -> str:
        response = self.chat.completions.create(
            model=model,
            messages=GPTClient._generate_prompt(scene_desc, fg_object)
        )
        return response.choices[0].message.content

    @staticmethod
    def _generate_prompt(scene_desc: str, fg_object: str) -> List[Dict[str, str]]:
        user_prompt["content"] = user_message.format(scene_desc=scene_desc, fg_object=fg_object)
        return messages + [user_prompt]


