# Knowledge Corpus Manifest

This document records the curated public-document expansion for the persistent RAG knowledge base.

## 2026-05-29 Expansion

Goal: expand the persistent RAG corpus from 7 real PDFs to 30 real PDFs with authoritative data-center HVAC, cooling, energy-efficiency, metering, and operations references.

Selection rule: prefer official or technically grounded sources from ASHRAE, U.S. DOE/FEMP, LBNL/Data Centers Center of Expertise, OCP, Uptime Institute, Google, and peer-reviewed/preprint research already present in the project. Downloaded but less central training/overview materials were kept in `data/knowledge_candidates` and not indexed into the final 30-document corpus.

## Newly Indexed Documents

| File | Authority | Topic | Source |
| --- | --- | --- | --- |
| `lbnl_guidance_zombie_servers_2024.pdf` | LBNL/DOE Data Centers Center of Expertise | Server utilization | https://datacenters.lbl.gov/sites/default/files/2024-07/Guidance%20on%20Finding%20Zombie%20Servers%20in%20Data%20Centers.pdf |
| `lbnl_it_tool_user_manual_2023.pdf` | LBNL/DOE Data Centers Center of Expertise | IT energy assessment | https://datacenters.lbl.gov/sites/default/files/2023-11/FINAL%20IT%20Tool%20User%20Manual%2010-27-2023%20Version%202.5.pdf |
| `lbnl_sandia_holistic_data_center_design_2023.pdf` | LBNL/DOE Data Centers Center of Expertise | Holistic design | https://datacenters.lbl.gov/sites/default/files/2023-02/Sandia%20Labs_Holistic%20Data%20Center%20Design.pdf |
| `lbnl_computer_server_selection_guidelines_2022.pdf` | LBNL/DOE Data Centers Center of Expertise | Server selection | https://datacenters.lbl.gov/sites/default/files/2022-03/Computer%20Server%20Selection%20Guidelines%2012-22%20%284%29.pdf |
| `lbnl_accessing_onboard_server_data_2021.pdf` | LBNL/DOE Data Centers Center of Expertise | Server telemetry | https://datacenters.lbl.gov/sites/default/files/FINAL%20Accessing%20Onboard%20Server%20Data%209-20-2021%20%281%29.pdf |
| `lbnl_dcee_actions_master_list_2020.pdf` | LBNL/DOE Data Centers Center of Expertise | Efficiency actions | https://datacenters.lbl.gov/sites/default/files/2024-04/DCEE%20Actions%20Master%20List_090920_final.pdf |
| `lbnl_data_center_resilience_demand_response_microgrids_2019.pdf` | LBNL/DOE Data Centers Center of Expertise | Resilience and demand response | https://datacenters.lbl.gov/sites/default/files/Designing%20and%20Managing%20Data%20Centers%20for%20Resilience%20-%20Demand%20Response%20and%20Microgrids_3Dec2019_0.pdf |
| `lbnl_liquid_cooling_new_horizons_2019.pdf` | LBNL/DOE Data Centers Center of Expertise | Liquid cooling | https://datacenters.lbl.gov/sites/default/files/7x24LiquidCooling_New_Horizons.pdf |
| `lbnl_getting_started_energy_efficiency_opportunities_2019.pdf` | LBNL/DOE Data Centers Center of Expertise | Energy assessment | https://datacenters.lbl.gov/sites/default/files/EEx%202019%20-%20Getting%20Started_%20Identifying%20and%20Assessing%20Energy%20Efficiency%20Opportunities%20in%20Your%20Data%20Center_final.pdf |
| `lbnl_open_spec_liquid_cooled_server_rack_2018.pdf` | LBNL/DOE Data Centers Center of Expertise | Liquid-cooled rack specification | https://datacenters.lbl.gov/sites/default/files/OpenSpecification.pdf |
| `lbnl_us_data_center_energy_usage_report_2016.pdf` | LBNL/DOE Data Centers Center of Expertise | U.S. data-center energy use | https://datacenters.lbl.gov/sites/default/files/DataCenterEnergyReport2016_0_0.pdf |
| `lbnl_system_tools_carbon_reduction_2022.pdf` | LBNL/DOE Data Centers Center of Expertise | Carbon-reduction tools | https://datacenters.lbl.gov/sites/default/files/2023-02/Training-%20System%20Tools%20with%20Carbon%20Reduction%203-23-2022_1.pdf |
| `lbnl_air_management_tool_user_manual_2023.pdf` | LBNL/DOE Data Centers Center of Expertise | Air-management tool | https://datacenters.lbl.gov/sites/default/files/2023-07/DOE%20AM%20Tool--User%27s%20Manual%20v3.1%20%287-25-2023%29.pdf |
| `lbnl_air_management_tool_engineering_reference_2023.pdf` | LBNL/DOE Data Centers Center of Expertise | Air-management engineering | https://datacenters.lbl.gov/sites/default/files/2023-03/DOE%20AM%20Tool--Eng%20Reference%20v3%20%283-15-2023%29.pdf |
| `lbnl_data_center_air_management_report_2006.pdf` | LBNL/DOE Data Centers Center of Expertise | Air management | https://datacenters.lbl.gov/sites/default/files/DC%20AIr%20Management%20Report_2006.pdf |
| `lbnl_air_management_small_data_centers_2016.pdf` | LBNL/DOE Data Centers Center of Expertise | Small data-center air management | https://datacenters.lbl.gov/sites/default/files/pgande_final_report_2016_herrlin2001204.pdf |
| `doe_data_center_energy_efficiency_fact_sheet_2025.pdf` | U.S. Department of Energy FEMP | Energy efficiency | https://www.energy.gov/sites/default/files/2025-08/femp-data-centers-fact-sheet-2025.pdf |
| `doe_thermosyphon_hybrid_cooling_water_efficiency_2019.pdf` | U.S. Department of Energy FEMP | Water-efficient cooling | https://www.energy.gov/sites/default/files/2019/05/f63/data-center-water-efficiency-0.pdf |
| `doe_hpc_energy_efficiency_opportunities_2013.pdf` | U.S. Department of Energy FEMP | HPC energy efficiency | https://www.energy.gov/cmei/femp/articles/energy-efficiency-opportunities-federal-high-performance-computing-data-centers |
| `doe_hpc_data_center_metering_protocol_2011.pdf` | U.S. Department of Energy FEMP | Metering protocol | https://www.energy.gov/cmei/femp/articles/high-performance-computing-data-center-metering-protocol |
| `ashrae_tc99_thermal_guidelines_refcard_2021.pdf` | ASHRAE TC 9.9 | Thermal guidelines | https://www.ashrae.org/file%20library/technical%20resources/bookstore/supplemental%20files/therm-gdlns-5th-r-e-refcard.pdf |
| `ashrae_904_2022_fact_sheet.pdf` | ASHRAE | Standard 90.4 | https://www.ashrae.org/file%20library/about/government%20affairs/advocacy%20toolkit/virtual%20packet/standard-90.4-2022-fact-sheet.pdf |
| `lbnl_ashrae_liquid_cooling_guidelines_hpc_2011.pdf` | LBNL/ASHRAE | Liquid-cooling guidelines | https://datacenters.lbl.gov/sites/default/files/ashrae-recommend-6-2011.pdf |

## Candidate Documents Not Indexed

These were downloaded and retained under `data/knowledge_candidates`, but not added to the final 30-document persistent corpus to avoid overshooting the requested target count or diluting the core HVAC/RAG signal:

- `lbnl_coe_dcoi_fact_sheet_2019.pdf`
- `lbnl_barriers_to_data_center_efficiency_2023.pdf`
- `lbnl_tour_coe_toolkit_resources_2023.pdf`
- `lbnl_assessment_tips_dcep_training_2023.pdf`
- `doe_federal_data_center_dashboard_recommendations_2014.pdf`
- `doe_nsidc_energy_reduction_strategies_2013.pdf`
- `doe_retro_commissioning_data_center_efficiency_2013.pdf`

## Verification Snapshot

- Persistent corpus before expansion: 7 documents, 340 chunks.
- Persistent corpus after expansion: 30 documents, 1,682 chunks.
- Embedding/index backend: `sentence-transformers`, `BAAI/bge-small-zh-v1.5`, FAISS.
- PDF validation: all 30 downloaded candidates had a `%PDF-` header and were readable with `pypdf`.
- Deduplication: no selected candidate matched an existing uploaded PDF by SHA256.
