import meta_sam_parser

print("Available parser functions:")
for name in dir(meta_sam_parser):
    if "parse" in name or "decode" in name:
        print(name)
