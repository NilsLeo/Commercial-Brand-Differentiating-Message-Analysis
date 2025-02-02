from dataclasses import dataclass

@dataclass
class BrandColors:
    primary: str = "#FF5733"
    secondary: str = "#33FF57"
    accent: str = "#3357FF"
    background: str = "#F0F0F0"
    text: str = "#333333"

# Usage
colors = BrandColors()
print(colors.primary)  # Output: #FF5733