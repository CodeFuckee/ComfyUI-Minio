from openai import OpenAI
import json


class OpenAIAPI:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "key": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "host": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)

    def main(self, data, key, host, model):

        client = OpenAI(api_key=key, base_url=host)
        import json
        messages = json.loads(data)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )

        print(response.choices[0].message.content)
        return (response.choices[0].message.content,)
