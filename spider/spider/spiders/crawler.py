import re

a = "Google and Facebook Divide Up Your Advertising Viewing  of the market into two empires.Google & Facebook's Share of U.S. Digital Ad Dollars68%In Alphabet Land, Google's role is the advertising gateway to the world. Google sells ads on its own digital"
print " ".join(re.findall(r"\w+", a))