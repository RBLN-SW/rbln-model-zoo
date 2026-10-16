import asyncio
import json

import numpy as np
import tritonclient.grpc.aio as grpcclient


# Define a simple chat message class
class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


# Apply a simple chat template to the messages
def apply_chat_template(messages):
    lines = []
    system_msg = ChatMessage(role="system", content="You are a helpful assistant.")
    for msg in [system_msg, *messages, ChatMessage(role="assistant", content="")]:
        lines.append(f"[{msg.role.capitalize()}]\n{msg.content}")
    return "\n".join(lines)


async def try_request():
    url = "localhost:8001"  # host and port number of the triton inference server
    client = grpcclient.InferenceServerClient(url=url, verbose=False)

    model_name = "vllm_model"

    def create_request(messages, request_id):
        prompt = apply_chat_template(messages)
        print(f"prompt:\n{prompt}\n---")  # print prompt

        input = grpcclient.InferInput("text_input", [1], "BYTES")
        prompt_data = np.array([prompt.encode("utf-8")])
        input.set_data_from_numpy(prompt_data)

        stream_setting = grpcclient.InferInput("stream", [1], "BOOL")
        stream_setting.set_data_from_numpy(np.array([True]))

        sampling_params = {
            "temperature": 0.0,
            "stop": ["[User]", "[System]", "[Assistant]"],  # add stop tokens
        }

        sampling_parameters = grpcclient.InferInput("sampling_parameters", [1], "BYTES")
        sampling_parameters.set_data_from_numpy(
            np.array([json.dumps(sampling_params).encode("utf-8")], dtype=object)
        )

        inputs = [input, stream_setting, sampling_parameters]
        output = grpcclient.InferRequestedOutput("text_output")
        outputs = [output]

        return {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": request_id,
        }

    messages = [
        ChatMessage(
            role="user", content="What is the first letter of English alphabets?"
        )
    ]
    request_id = "req-0"

    async def requests_gen():
        yield create_request(messages, request_id)

    response_stream = client.stream_infer(requests_gen())

    prompt = apply_chat_template(messages)
    is_first_response = True

    async for response in response_stream:
        result, error = response
        if error:
            print("Error occurred!")
        else:
            output = result.as_numpy("text_output")
            for i in output:
                decoded = i.decode("utf-8")
                if is_first_response:
                    if decoded.startswith(prompt):
                        decoded = decoded[len(prompt) :]
                    is_first_response = False
                print(decoded, end="", flush=True)
    print("\n")  # end of stream


asyncio.run(try_request())
