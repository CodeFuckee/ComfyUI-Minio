from .shared import is_cn


class IsTextZhCN:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "-1",
                    },
                ),
            },
        }

    CATEGORY = "ComfyUI-Minio"
    FUNCTION = "main"
    RETURN_TYPES = ("BOOLEAN",)

    def main(self, text):
        return (is_cn(text),)
