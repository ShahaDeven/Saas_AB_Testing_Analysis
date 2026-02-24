"""
SaaS A/B Test Data Simulator
============================

Generates a realistic dataset for A/B testing analysis with:
- Known treatment effects (so you can verify your analysis recovers them)
- Segment-level heterogeneity (different effects for different user types)
- Simpson's Paradox built-in (overall effect differs from segment effects)
- Temporal patterns (realistic signup distribution)

"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Set seed for reproducibility
np.random.seed(42)


def generate_ab_test_data(
    n_users: int = 50000,
    test_start_date: str = "2025-01-06",
    test_duration_days: int = 28,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate simulated A/B test data for a SaaS onboarding experiment.
    
    The Data Generating Process (DGP):
    ----------------------------------
    - Users are randomly assigned to control (50%) or treatment (50%)
    - Baseline activation rates vary by company_size:
        - SMB: 25% baseline
        - Mid-market: 35% baseline  
        - Enterprise: 45% baseline
    - Treatment effects vary by company_size (HETEROGENEOUS TREATMENT EFFECT):
        - SMB: +8% lift (guided onboarding helps confused users)
        - Mid-market: +4% lift (moderate benefit)
        - Enterprise: -2% lift (they prefer self-serve, wizard is annoying)
    - This creates SIMPSON'S PARADOX:
        - SMB is 60% of users, enterprise is 10%
        - Overall effect is positive, but enterprise effect is negative
    
    Parameters
    ----------
    n_users : int
        Total number of users to simulate
    test_start_date : str
        Start date of the experiment (YYYY-MM-DD)
    test_duration_days : int
        Length of the experiment in days
    output_path : str, optional
        If provided, saves the dataset to this path
        
    Returns
    -------
    pd.DataFrame
        Simulated A/B test dataset
    """
    
    print(f"Generating {n_users:,} users for A/B test simulation...")
    
    # ===================
    # 1. USER ATTRIBUTES
    # ===================
    
    # User IDs
    user_ids = [f"user_{i:06d}" for i in range(1, n_users + 1)]
    
    # Signup dates (with weekday pattern - more signups on weekdays)
    start_date = datetime.strptime(test_start_date, "%Y-%m-%d")
    dates = []
    for _ in range(n_users):
        day_offset = np.random.randint(0, test_duration_days)
        signup_date = start_date + timedelta(days=day_offset)
        # Reduce weekend signups by 40%
        if signup_date.weekday() >= 5:  # Saturday=5, Sunday=6
            if np.random.random() < 0.4:
                # Shift to nearest weekday
                day_offset = (day_offset + 2) % test_duration_days
                signup_date = start_date + timedelta(days=day_offset)
        dates.append(signup_date)
    
    # Variant assignment (50/50 split - true randomization)
    variants = np.random.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])
    
    # Company size distribution (creates Simpson's Paradox)
    # SMB dominates, which is realistic for SaaS
    company_sizes = np.random.choice(
        ["smb", "mid_market", "enterprise"],
        size=n_users,
        p=[0.60, 0.30, 0.10] 
    )
    
    plan_types = []
    for size in company_sizes:
        if size == "smb":
            plan = np.random.choice(["free_trial", "paid"], p=[0.75, 0.25])
        elif size == "mid_market":
            plan = np.random.choice(["free_trial", "paid"], p=[0.50, 0.50])
        else:  # enterprise
            plan = np.random.choice(["free_trial", "paid"], p=[0.20, 0.80])
        plan_types.append(plan)

    traffic_sources = np.random.choice(
        ["organic", "paid_ads", "referral"],
        size=n_users,
        p=[0.50, 0.35, 0.15]
    )
    
    devices = []
    for size in company_sizes:
        if size == "enterprise":
            device = np.random.choice(["desktop", "mobile"], p=[0.95, 0.05])
        else:
            device = np.random.choice(["desktop", "mobile"], p=[0.80, 0.20])
        devices.append(device)
    
    # ===================
    # 2. ACTIVATION OUTCOMES (THE KEY PART)
    # ===================
    
    # Define baseline activation rates and treatment effects by segment
    # This is the "ground truth" you should recover in your analysis
    BASELINE_RATES = {
        "smb": 0.25,
        "mid_market": 0.35,
        "enterprise": 0.45
    }
    
    # TREATMENT EFFECTS (absolute percentage point change)
    # This creates heterogeneous treatment effects
    TREATMENT_EFFECTS = {
        "smb": 0.08,        # +8pp lift (big help)
        "mid_market": 0.04, # +4pp lift (moderate help)
        "enterprise": -0.02 # -2pp lift (slight harm - they don't like hand-holding)
    }
    
    # Generate 7-day activation
    activated_7d = []
    for i in range(n_users):
        size = company_sizes[i]
        variant = variants[i]
        
        base_prob = BASELINE_RATES[size]
        
        # Add treatment effect if in treatment group
        if variant == "treatment":
            prob = base_prob + TREATMENT_EFFECTS[size]
        else:
            prob = base_prob
        
        if plan_types[i] == "paid":
            prob += 0.05 
        if traffic_sources[i] == "referral":
            prob += 0.03  
        if devices[i] == "mobile":
            prob -= 0.05 
        
        prob = np.clip(prob, 0.01, 0.99)
        
        # Bernoulli draw
        activated = np.random.random() < prob
        activated_7d.append(activated)
    
    # Generate 14-day activation (higher than 7-day, correlated)
    activated_14d = []
    for i in range(n_users):
        if activated_7d[i]:
            activated_14d.append(True)
        else:
            # Additional chance to activate between day 7-14
            # Treatment effect slightly smaller for late activators
            size = company_sizes[i]
            variant = variants[i]
            
            late_activation_prob = 0.10  # 10% of non-7d-activators activate by 14d
            if variant == "treatment":
                late_activation_prob += TREATMENT_EFFECTS[size] * 0.3  # Reduced effect
            
            late_activation_prob = np.clip(late_activation_prob, 0.01, 0.50)
            activated_14d.append(np.random.random() < late_activation_prob)
    
    # Generate actions completed (correlated with activation)
    actions_completed = []
    for i in range(n_users):
        if activated_14d[i]:
            # Activated users: 3-15 actions
            actions = np.random.poisson(7) + 3
        elif activated_7d[i]:
            # This shouldn't happen given our logic, but safety
            actions = np.random.poisson(5) + 2
        else:
            # Non-activated: 0-3 actions
            actions = np.random.poisson(1)
        actions_completed.append(min(actions, 30))  # Cap at 30
    
    # ===================
    # 3. BUILD DATAFRAME
    # ===================
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "signup_date": dates,
        "variant": variants,
        "company_size": company_sizes,
        "plan_type": plan_types,
        "traffic_source": traffic_sources,
        "device": devices,
        "actions_completed": actions_completed,
        "activated_7d": activated_7d,
        "activated_14d": activated_14d
    })

    df = df.sort_values("signup_date").reset_index(drop=True)
    
    # ===================
    # 4. PRINT SUMMARY
    # ===================
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total users: {len(df):,}")
    print(f"Date range: {df['signup_date'].min().date()} to {df['signup_date'].max().date()}")
    print(f"\nVariant split:")
    print(df["variant"].value_counts())
    print(f"\nCompany size distribution:")
    print(df["company_size"].value_counts(normalize=True).round(3))
    
    print("\n" + "-"*60)
    print("GROUND TRUTH TREATMENT EFFECTS (what your analysis should find)")
    print("-"*60)
    for size in ["smb", "mid_market", "enterprise"]:
        baseline = BASELINE_RATES[size]
        effect = TREATMENT_EFFECTS[size]
        print(f"  {size:12s}: baseline={baseline:.0%}, treatment effect={effect:+.0%}")
    
    print("\n" + "-"*60)
    print("OBSERVED ACTIVATION RATES (7-day)")
    print("-"*60)
    
    summary = df.groupby(["company_size", "variant"])["activated_7d"].mean().unstack()
    summary["lift"] = summary["treatment"] - summary["control"]
    print(summary.round(4))
    
    overall = df.groupby("variant")["activated_7d"].mean()
    print(f"\nOVERALL: control={overall['control']:.4f}, treatment={overall['treatment']:.4f}, lift={overall['treatment']-overall['control']:.4f}")
    
    # ===================
    # 5. SAVE TO FILE
    # ===================
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Dataset saved to: {output_path}")
    
    return df


def print_ground_truth():
    """Print the ground truth for reference during analysis."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    GROUND TRUTH (SPOILERS)                   ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Segment      │ Baseline │ Treatment Effect │ Direction      ║
    ╠═══════════════╪══════════╪══════════════════╪════════════════╣
    ║  SMB          │   25%    │      +8pp        │ ✓ Positive     ║ 
    ║  Mid-market   │   35%    │      +4pp        │ ✓ Positive     ║
    ║  Enterprise   │   45%    │      -2pp        │ ✗ Negative     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  SIMPSON'S PARADOX: Overall effect is positive because SMB   ║
    ║  (60% of users) has strong positive effect, masking the      ║
    ║  negative effect on enterprise (10% of users).               ║
    ║                                                              ║
    ║  RECOMMENDATION: Ship to SMB and mid-market only.            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    output_path = "data/raw/ab_test_data.csv"
    
    df = generate_ab_test_data(
        n_users=50000,
        test_start_date="2025-01-06",
        test_duration_days=28,
        output_path=output_path
    )
    
    print("\n")
    print_ground_truth()
