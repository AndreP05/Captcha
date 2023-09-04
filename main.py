import CapBreak

"""

Teste com Captcha Real

"""

mode = 0

if mode:
    x = CapBreak.break_captcha("example.jpg", mode=mode)
    result = ""

    for i in x:
        result += str(int(i[0]))

    if result == "19973":
        CapBreak.register_success(result)
else:
    x = CapBreak.break_captcha("example.jpg")
    print(x)
