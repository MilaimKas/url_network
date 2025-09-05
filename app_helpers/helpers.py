import pandas as pd
import ast
import numpy as np

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val  # fallback if not a list

def load_and_fix_csv(path, list_cols = ['url_path', 'chrono', 'last_node']):
    df = pd.read_csv(path)

    for col in list_cols:
        df[col] = df[col].apply(safe_literal_eval)

    # Optional: convert float64-like strings to actual floats
    def flatten_np_float(x):
        if isinstance(x, list):
            return [float(str(t).replace("np.float64(", "").replace(")", "")) for t in x]
        return x

    df['chrono'] = df['chrono'].apply(flatten_np_float)

    return df

def safe_float(x, default=0.0):
    """Cleans floats for JSON serialization."""
    try:
        f = float(x)
        if not np.isfinite(f):
            return default
        return f
    except:
        return default

url_sample = [
            "home",
            "home/about",
            "home/contact",
            "home/partners",
            "home/faq",

            "home/legal",
            "home/legal/privacy-policy",
            "home/legal/terms-of-service",

            "home/products",
            "home/products/pricing",
            "home/products/comparison",
            "home/products/features",
            "home/products/features/integration",
            "home/products/features/security",
            "home/products/features/ai-tools",
            "home/products/demo",
            "home/products/download",

            "home/blog",
            "home/blog/categories",
            "home/blog/categories/technology",
            "home/blog/categories/business",
            "home/blog/tags",
            "home/blog/tags/ai",
            "home/blog/tags/security",
            "home/blog/2020",
            "home/blog/2021",
            "home/blog/2022",
            "home/blog/2023",

            "home/press",
            "home/press/2020",
            "home/press/2021",
            "home/press/2022",
            "home/press/2023",
            "home/press/media-kit",

            "home/careers",
            "home/careers/open-positions",
            "home/careers/internships",
            "home/careers/remote",
            "home/careers/europe",
            "home/careers/usa",
            "home/careers/culture",

            "home/support",
            "home/support/faq",
            "home/support/contact",
            "home/support/ticket",
            "home/support/guides",
            "home/support/tutorials",
            "home/support/tutorials/getting-started",
            "home/support/tutorials/advanced-features",

            "home/login",
            "home/signup",
            "home/forgot-password",
            "home/reset-password",

            "home/account",
            "home/account/profile",
            "home/account/settings",
            "home/account/notifications",
            "home/account/security",
            "home/account/billing",
            "home/account/subscription",
            "home/account/teams",

            "home/dashboard",
            "home/dashboard/overview",
            "home/dashboard/analytics",
            "home/dashboard/reports",
            "home/dashboard/exports",

            "home/integrations",
            "home/integrations/google",
            "home/integrations/slack",
            "home/integrations/stripe",
            "home/integrations/salesforce",

            "home/locales/en",
            "home/locales/fr",
            "home/locales/de",
            "home/locales/es",
            "home/locales/it",
        ]
