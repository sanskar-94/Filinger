# The analyst agent on real filings (nvidia/nemotron-3.5-lightning-30b-a3b, BM25)

10 questions about real companies, asked of the lake built from their EDGAR filings (`data/real_pilot/lake`). There is no answer key: each answer is followed by the full text of every paragraph it cites, so it can be judged against its sources. Market reactions come from the event table (the 3 day abnormal return against SPY).

## 1. Did Walmart raise, lower or keep its full year guidance in the earnings release it filed in August 2024?

Company in focus: WMT. Turns: 3. Tool calls: search_filings, get_paragraph.

Walmart **raised** its full-year 2025 guidance in the August 15, 2024 earnings release (8-K accession 0000104169‑24‑000131). The release presented both the “updated” and “original” FY 2025 outlook, and every major metric was increased:

- **Consolidated net sales (cc):** 3.75%–4.75% (up from 3.0%–4.0%)  
- **Consolidated adjusted operating income (cc):** 6.5%–8.0% (up from 4.0%–6.0%)  
- **Adjusted EPS:** $2.35–$2.43 (up from $2.23–$2.37)  

[cite:1f80c3fa02591ef9]

> **[cite:1f80c3fa02591ef9]** WMT 8-K filed 2024-08-15, section `8k_results_of_operations`, accession 0000104169-24-000131
>
> Fiscal Year 2025 The Company's fiscal year guidance is based on the following FY24 figures: Net sales: $642.6 billion, adjusted operating income1: $27.1 billion, and adjusted EPS1: $2.22. The Company's full year guidance assumes a generally stable consumer and continued pressure from its mix of products and formats globally. Metric FY 2025 (updated) FY 2025 (original) Consolidated net sales (cc) Increase 3.75% to 4.75% Increase 3.0% to 4.0% Consolidated adj. operating income (cc) Increase 6.5% to 8.0% Increase 4.0% to 6.0% Interest, net Increase approximately $100M Increase approximately $100M to $200M Effective tax rate Lower-end of original guidance Approximately 25.0% to 26.0% Non-controlling interest Unchanged from original guidance Relatively flat Adjusted EPS $2.35 to $2.43 $2.23 to $2.37 Capital expenditures Unchanged from original guidance Approximately 3.0% to 3.5% of net sales 1 For relevant reconciliations, see Q3 FY24 and Q4 FY24 earnings releases furnished on Form 8-K on November 16, 2023 and February 20, 2024, respectively. Per share amounts have been retroactively adjusted to reflect the February 23, 2024 stock split.

## 2. How did the stock react to the earnings release Nvidia filed in May 2024?

Company in focus: NVDA. Turns: 3. Tool calls: list_filings, get_event_return.

**Stock reaction to Nvidia’s May 2024 earnings release**

- On the first tradable day after the 8‑K (filed 22 May 2024), NVDA’s price rose **≈ 9.3 %** (day‑0 return).  
- Over the following three trading days the cumulative abnormal return (CAR) was **≈ 18.9 %** versus the market (CAR 3d = 0.1887) or **≈ 18.1 %** when adjusted for the stock’s beta (CAR 3d market‑model = 0.1808).  

These figures come from the event‑return analysis of the 8‑K accession **0001045810‑24‑000113** ([market data]). The positive market response indicates that investors viewed the quarterly results favorably.

## 3. Did ExxonMobil agree to acquire Pioneer Natural Resources, and on what terms?

Company in focus: XOM. Turns: 2. Tool calls: search_filings.

Yes. On **October 11 2023** ExxonMobil and Pioneer Natural Resources jointly announced a definitive merger agreement under which ExxonMobil will acquire Pioneer in an **all‑stock transaction valued at $59.5 billion (≈ $253 per share)**[cite:1cf3e4e2f55ac764].  

The deal **closed on May 3 2024**. In the subsequent 8‑K filing (August 2 2024) ExxonMobil reported that it issued **545 million shares of its common stock having a fair value of $63 billion on the acquisition date and assumed $5 billion of debt**[cite:414657fb0ca0a81c].  

**Terms summary**

| Item | Detail |
|------|--------|
| **Acquirer** | ExxonMobil (ticker XOM) |
| **Target** | Pioneer Natural Resources (ticker PXD) |
| **Transaction type** | All‑stock merger |
| **Enterprise value** | $59.5 billion |
| **Per‑share price** | $253 (based on ExxonMobil’s share price at announcement) |
| **Completion date** | May 3 2024 |
| **Consideration at close** | 545 M ExxonMobil shares (fair value $63 B) + assumption of $5 B debt |

Thus, ExxonMobil did agree to acquire Pioneer Natural Resources, and the agreed terms were an all‑stock deal worth $59.5 billion at $253 per share, later completed with the issuance of 545 million ExxonMobil shares and $5 billion of assumed debt.

> **[cite:1cf3e4e2f55ac764]** XOM 8-K filed 2023-10-11, section `8k_press_release`, accession 0000950103-23-014885
>
> U.S. economy and energy security SPRING, Texas & IRVING, Texas – Exxon Mobil Corporation (NYSE: XOM) and Pioneer Natural Resources (NYSE: PXD) jointly announced a definitive agreement for ExxonMobil to acquire Pioneer. The merger is an all-stock transaction valued at $59.5 billion, or $253 per share, based on

> **[cite:414657fb0ca0a81c]** XOM 8-K filed 2024-08-02, section `8k_results_of_operations`, accession 0000034088-24-000048
>
> •On May 3, 2024, ExxonMobil completed the acquisition of Pioneer Natural Resources. The company issued 545 million shares of ExxonMobil common stock having a fair value of $63 billion on the acquisition date and assumed $5 billion of debt. The merger of ExxonMobil and Pioneer created the largest, high-return unconventional resource development potential in the world.

## 4. What did JPMorgan disclose about its acquisition of First Republic Bank?

Company in focus: JPM. Turns: 3. Tool calls: search_filings, get_paragraph.

JPMorgan Chase disclosed that on **May 1, 2023** it acquired certain assets and assumed certain liabilities of **First Republic Bank** from the **FDIC**, resulting in an **estimated bargain‑purchase gain of $2.7 billion** recorded in other income. All subsequent references to "excluding First Republic" or "attributable to First Republic" refer to the effects of that acquisition [cite:e1e317b1976e1d6b].

> **[cite:e1e317b1976e1d6b]** JPM 8-K filed 2023-07-14, section `8k_results_of_operations`, accession 0000019617-23-000425
>
> 8.On May 1, 2023, JPMorgan Chase acquired certain assets and assumed certain liabilities of First Republic Bank (the "First Republic acquisition") from the Federal Deposit Insurance Corporation ("FDIC") resulting in an estimated bargain purchase gain of $2.7 billion recorded in other income. All references in this press release to "excluding First Republic" or "attributable to First Republic" refer to excluding or including, as applicable, the relevant effects of the First Republic acquisition. In the second quarter of 2023, expense attributable to First Republic was substantially all in Corporate.

## 5. What has Boeing disclosed about the Alaska Airlines Flight 1282 accident?

Company in focus: BA. Turns: 6. Tool calls: search_filings, get_paragraph, get_paragraph, get_paragraph, get_paragraph.

**Boeing’s key disclosures about the Alaska Airlines Flight 1282 accident (737‑9 door‑plug event)**  

- **Cooperation & independent review** – Boeing “continues to cooperate transparently with the FAA” and has “appointed an outside expert to lead an in‑depth independent assessment of Commercial Airplanes’ quality‑management system,” with recommendations sent directly to the CEO and the Aerospace Safety Committee [cite:b56f8f997c026076].  

- **Immediate quality‑strengthening actions** – Requiring additional inspections inside Boeing’s factory and at key suppliers, supporting expanded airline‑customer oversight, and pausing 737 production for one day to refocus employees on quality [cite:b56f8f997c026076].  

- **Production‑rate changes** – Prior to the accident Boeing was running at 38 units/month. In Q1 2024 it “slowed production rates to reduce traveled work in our factory, as well as at our suppliers,” and is now conducting fuselage inspections at Spirit (its final‑assembly site) before shipment [cite:7260dba3bed6e5b9].  

- **Financial impact** – The accident and the resulting actions (including the production slowdown) “have significantly impacted our financial position, results of operations and cash flows during the first half of 2024” [cite:da054e55ce597edb].  

- **Inspection & fleet‑return process** – The FAA grounded all 737‑9s with a mid‑exit door plug and required an enhanced inspection. On Jan 24 2024 the FAA approved that process; by the first quarter all grounded 737‑9 operators had returned their fleets to service, and every 737‑9 in production is now undergoing the enhanced inspection before delivery [cite:194a2ea80ca91f41].  

These points capture the main factual and operational disclosures Boeing has made about the accident across its 8‑K (Jan 31 2024) and three 10‑Q filings (April 24, July 31, and Oct 23 2024).

> **[cite:b56f8f997c026076]** BA 8-K filed 2024-01-31, section `8k_results_of_operations`, accession 0000012927-24-000007
>
> The company continues to cooperate transparently with the FAA following the Alaska Airlines Flight 1282 accident involving a 737-9. Commercial Airplanes is taking immediate actions to strengthen quality on the 737 program, including requiring additional inspections within its factory and at key suppliers, supporting expanded oversight from airline customers and pausing 737 production for one day to refocus its employees on quality. The company has also appointed an outside expert to lead an in-depth independent assessment of Commercial Airplanes' quality management system, with recommendations provided directly to Calhoun and the Aerospace Safety Committee of Boeing's Board of Directors.

> **[cite:7260dba3bed6e5b9]** BA 10-Q filed 2024-04-24, section `mdna`, accession 0000012927-24-000025
>
> Our planned production rates are dependent on our suppliers' ability to support our operations and our ability to meet heightened quality control requirements. Prior to the Alaska Airlines accident, we were operating at a production rate of 38 per month. During the first quarter of 2024, as part of our plan to address the issues identified, we slowed production rates to reduce traveled work in our factory, as well as at our suppliers. We are now conducting fuselage inspections at Spirit to ensure quality prior to shipment to Boeing.

> **[cite:da054e55ce597edb]** BA 10-Q filed 2024-07-31, section `mdna`, accession 0000012927-24-000055
>
> The Alaska Airlines accident and the resulting actions we are taking, including slowing production, to improve compliance with our manufacturing quality control requirements have significantly impacted our financial position, results of operations and cash flows during the first half of 2024.

> **[cite:194a2ea80ca91f41]** BA 10-Q filed 2024-10-23, section `mdna`, accession 0000012927-24-000082
>
> Overview On January 5, 2024, an Alaska Airlines 737-9 flight made an emergency landing after a mid-exit door plug detached in flight. Following the accident, the Federal Aviation Administration (FAA) grounded and required inspections of all 737-9 aircraft with a mid-exit door plug, which constituted the large majority of the approximately 220 737-9 aircraft in the in-service fleet. On January 24, 2024, the FAA approved an enhanced maintenance and inspection process that was required to be performed on each of the grounded 737-9 aircraft. Our 737-9 operators returned their fleets to service in the first quarter. All 737-9 aircraft in production are undergoing this same enhanced inspection process prior to delivery.

## 6. What did Pfizer say about revenue from its COVID-19 products in 2023?

Company in focus: PFE. Turns: 6. Tool calls: search_filings, get_paragraph, get_paragraph, get_paragraph, search_filings.

Pfizer’s 2023 COVID‑19 product revenue guidance evolved through the year:

- **Initial 2023 guidance (8‑K, Jan 31 2023)** – Comirnaty ≈ $13.5 B and Paxlovid ≈ $8 B [cite:feb81e4c2eadaff6].  
- **Mid‑year update (8‑K, Oct 13 2023)** – Total COVID‑19 product revenue expected at ~$12.5 B, a $9 B reduction from the earlier outlook; Comirnaty alone was pegged at ~$11.5 B [cite:eb4e3c993694f2d9].  
- **Quarter‑end update (8‑K, Oct 31 2023)** – Comirnaty revenues of about $11.5 B, down ~70 % from 2022; the overall revenue decline for 2023 was attributed primarily to expected falls in COVID‑19 product sales [cite:05ccc0e5cdb1ff05][cite:0284ca4723becfef].  

In summary, Pfizer began 2023 expecting roughly $13.5 B from Comirnaty and $8 B from Paxlovid, then revised the combined COVID‑19 product total down to about $12.5 B by October, with Comirnaty contributing ~$11.5 B and Paxlovid the remainder.

> **[cite:feb81e4c2eadaff6]** PFE 8-K filed 2023-01-31, section `8k_results_of_operations`, accession 0000078003-23-000004
>
> –Full-Year 2023 Revenue Guidance for Comirnaty(1) of ~$13.5 Billion and Paxlovid of ~$8 Billion

> **[cite:eb4e3c993694f2d9]** PFE 8-K filed 2023-10-13, section `8k_results_of_operations`, accession 0000078003-23-000099
>
> Updates Full-Year 2023 Revenue and Adjusted(2) Diluted EPS Guidance(1) Ranges Pfizer also announced that it now anticipates full-year 2023 revenues to be in the range of $58.0 to $61.0 billion, versus its previous guidance range of $67.0 to $70.0 billion solely due to its COVID products. Full-year 2023 revenues for Paxlovid and Comirnaty are expected to be approximately $12.5 billion, a decline of $9.0 billion versus original expectations. The company is reducing its full-year 2023 revenue expectations for Paxlovid by approximately $7.0 billion which includes a non-cash $4.2 billion revenue reversal for the return of the 7.9 million treatment courses of EUA-labeled U.S. government inventory, as well as the delayed commercialization to January 2024 versus our previous expectation of commercialization in the second half of 2023. The company is also reducing its full-year 2023 revenue expectations for Comirnaty by approximately $2.0 billion due to lower-than-expected vaccination rates.

> **[cite:05ccc0e5cdb1ff05]** PFE 8-K filed 2023-10-31, section `8k_results_of_operations`, accession 0000078003-23-000111
>
> ▪Comirnaty(1) revenues of approximately $11.5 billion, down 70% from 2022 results.

> **[cite:0284ca4723becfef]** PFE 8-K filed 2023-10-31, section `8k_results_of_operations`, accession 0000078003-23-000111
>
> Excluding COVID-19 products, Pfizer is expecting 6% to 8% operational revenue growth in 2023. Revenue guidance for Pfizer's COVID-19 products is as follows:

## 7. Has Coca-Cola disclosed a tax dispute with the Internal Revenue Service?

Company in focus: KO. Turns: 2. Tool calls: search_filings.

Yes. Coca‑Cola has repeatedly disclosed tax disputes with the Internal Revenue Service in its SEC filings. Key disclosures include:

- In a 2021 10‑Q (paragraph **[cite:a98293103543f29d]**), the company reported that on September 17 2015 it received an IRS notice seeking approximately **$3.3 billion** of additional federal income tax for the years 2007 through 2009, involving the reallocation of over $9 billion of income to the U.S. parent company from foreign affiliates.  

- The same 10‑Q (paragraph **[cite:13c3c754915df946]**) states that on **October 15 2015** the IRS designated the company’s matter for **litigation**, which forecloses any alternative means of resolving the tax dispute and forces the company to either accept the IRS’ assessment or litigate in federal court. The matter remains under that litigation designation.

- A 2023 10‑K (paragraph **[cite:df711d900974f218]**) references a **Closing Agreement** reached in 1996 that resolved a transfer‑pricing dispute for the tax years 1987 through 1995, under which the IRS agreed not to assess penalties for using the agreed‑upon methodology going forward, absent a change in material facts or law.

These excerpts confirm that Coca‑Cola has disclosed an ongoing IRS tax dispute, including a large‑scale assessment and an IRS litigation designation that prevents settlement.

> **[cite:a98293103543f29d]** KO 10-Q filed 2021-04-27, section `legal`, accession 0000021344-21-000014
>
> U.S. Federal Income Tax Dispute On September 17, 2015, the Company received a Notice from the IRS seeking approximately $3.3 billion of additional federal income tax for years 2007 through 2009. In the Notice, the IRS stated its intent to reallocate over $9 billion of income to the U.S. parent company from certain of its foreign affiliates that the U.S. parent company licensed to manufacture, distribute, sell, market and promote its products in certain non-U.S. markets.

> **[cite:13c3c754915df946]** KO 10-Q filed 2024-10-24, section `legal`, accession 0000021344-24-000060
>
> The IRS designated the Company's matter for litigation on October 15, 2015. Litigation designation is an IRS determination that forecloses to a company any and all alternative means for resolution of a tax dispute. As a result of the IRS' designation of the Company's matter for litigation, the Company was forced to either accept the IRS' newly imposed tax assessment and pay the full amount of the asserted tax or litigate the matter in the federal courts. The matter remains subject to the IRS' litigation designation, preventing the Company from any attempt to settle or otherwise mutually resolve the matter with the IRS.

> **[cite:df711d900974f218]** KO 10-K filed 2023-02-21, section `legal`, accession 0000021344-23-000011
>
> compensation from its foreign licensees. The Company and the IRS memorialized this accord in a closing agreement resolving that dispute ("Closing Agreement"). The Closing Agreement provided that, absent a change in material facts or circumstances or relevant federal tax law, in calculating the Company's income taxes going forward, the Company would not be assessed penalties by the IRS for using the agreed-upon tax calculation methodology that the Company and the IRS agreed would be used for the 1987 through 1995 tax years.

## 8. What did Apple say drove the change in its gross margin in fiscal 2024?

Company in focus: AAPL. Turns: 5. Tool calls: search_filings, search_filings, get_paragraph, get_paragraph.

Apple’s 2024 10‑K attributes the overall gross‑margin change in fiscal 2024 to the following drivers:

- **Products gross margin** rose year‑over‑year **due to cost savings**, partially offset by a different product mix and foreign‑currency weakness relative to the U.S. dollar [cite:db5e0e2bed611189].  
- **Services gross margin** increased **primarily because of higher Services net sales** [cite:1e7ffbc047c61017].  

These two statements together explain what Apple identified as the key factors behind the gross‑margin shift in its fiscal‑2024 results.

> **[cite:db5e0e2bed611189]** AAPL 10-K filed 2024-11-01, section `mdna`, accession 0000320193-24-000123
>
> Products Gross Margin Products gross margin and Products gross margin percentage increased during 2024 compared to 2023 due to cost savings, partially offset by a different Products mix and the weakness in foreign currencies relative to the U.S. dollar.

> **[cite:1e7ffbc047c61017]** AAPL 10-K filed 2024-11-01, section `mdna`, accession 0000320193-24-000123
>
> Services Gross Margin Services gross margin increased during 2024 compared to 2023 due primarily to higher Services net sales.

## 9. Has Visa said there is substantial doubt about its ability to continue as a going concern?

Company in focus: V. Turns: 8. Tool calls: search_filings, search_filings, search_filings, search_filings, search_filings, search_filings, search_filings.

Based on the SEC filings searched for Visa (ticker V), there is no disclosure of substantial doubt about the company's ability to continue as a going concern. The searches through 10-K, 10-Q, and 8-K filings returned various risk factors (competition, regulation, cybersecurity, etc.) but no paragraph explicitly stating going concern doubt. The filings do not contain the required going concern disclosure.

## 10. Has Home Depot announced a new share repurchase authorization?

Company in focus: HD. Turns: 2. Tool calls: search_filings.

Yes. On **August 15 2023** Home Depot’s Board of Directors authorized a new **$15 billion share‑repurchase program**, replacing the prior authorization. This was disclosed in an 8‑K filing (accession 0000354950‑23‑000180, paragraph [81357460dc69efca]). A later 10‑Q (filed 2024‑11‑19) also references the August 2023 $15 billion authorization, noting that as of October 27 2024 about $11.7 billion remained available. Earlier authorizations include a $20 billion program approved in May 2021 and a $15 billion program approved in August 2022, but the most recent new announcement is the August 2023 $15 billion repurchase authorization.

