import uuid
import time
import schedule
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

def job():
    # ─── CONFIG ─────────────────────────────────────────────────────────
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Not directly used for Qdrant ingestion with SentenceTransformer
    COLLECTION_NAME = "FAQS_COLLCECTION"
    EMBED_MODEL_ID = "all-MiniLM-L6-v2"
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333

    # Hardcoded FAQ data (all 267 FAQs)
    FAQ_DATA = [
        {"question": "Q1: Can foreigners buy property in Dubai?", "answer": "Yes, foreigners can buy freehold properties in designated areas of Dubai like Downtown Dubai, Dubai Marina, Palm Jumeirah, and Jumeirah Village Circle (JVC)."},
        {"question": "Q2: What is the difference between freehold and leasehold in Dubai?", "answer": "Freehold allows full ownership of the property and land, while leasehold grants rights for 10 to 99 years without land ownership."},
        {"question": "Q3: What are the top freehold communities in Dubai for expats?", "answer": "Popular freehold areas include Dubai Hills Estate, Palm Jumeirah, Emaar South, Business Bay, and Dubai Creek Harbour."},
        {"question": "Q4: Which are the best towers in Dubai Marina for investment?", "answer": "Top-performing towers include Marina Gate, Cayan Tower, Princess Tower, and Ocean Heights."},
        {"question": "Q5: Can you own property in Downtown Dubai as a non-resident?", "answer": "Yes, Downtown Dubai is a freehold area where non-residents can purchase apartments."},
        {"question": "Q6: Can expats buy property in Abu Dhabi?", "answer": "Yes, expats can buy properties in designated investment zones such as Al Reem Island, Saadiyat Island, Yas Island, and Al Raha Beach."},
        {"question": "Q7: What types of ownership are available for expats in Abu Dhabi?", "answer": "Options include usufruct (right to use), musataha (right to develop), and full freehold in designated areas."},
        {"question": "Q8: Which are the most popular communities in Abu Dhabi for property investment?", "answer": "Yas Island, Al Reem Island, Al Raha Beach, and Saadiyat Island are top choices for both investors and residents."},
        {"question": "Q9: What are the best residential towers in Al Reem Island?", "answer": "Key towers include Sun & Sky Towers, The Gate Towers, and Ocean Terrace."},
        {"question": "Q10: What is the minimum property investment for a Golden Visa in Abu Dhabi?", "answer": "AED 2 million is the minimum investment required to qualify for a UAE Golden Visa through real estate."},
        {"question": "Q11: Can foreigners buy property in Sharjah?", "answer": "Yes, since 2014, non-Arab expats can buy property in selected projects on a 100-year leasehold basis, such as Aljada and Tilal City."},
        {"question": "Q12: What is the difference between property ownership in Sharjah and Dubai?", "answer": "In Sharjah, property is usually sold as leasehold to non-GCC nationals, whereas Dubai offers full freehold ownership in designated zones."},
        {"question": "Q13: What are the top communities to buy property in Sharjah?", "answer": "Popular areas include Aljada, Tilal City, Al Nahda, Muwaileh, and Maryam Island."},
        {"question": "Q14: Are there free zones in Sharjah where expats can invest in property?", "answer": "Sharjah does not offer traditional freehold zones but permits long-term leasehold in master-planned developments."},
        {"question": "Q15: What types of properties can you buy in Sharjah?", "answer": "Expats can invest in apartments, townhouses, and villas in leasehold areas like Aljada and Tilal City."},
        {"question": "Q16: What is the Golden Visa real estate requirement in the UAE?", "answer": "To obtain a 10-year UAE Golden Visa through real estate, the property must be worth at least AED 2 million, fully paid or under mortgage with a minimum of AED 2M equity."},
        {"question": "Q17: Is there a property transfer fee in Dubai?", "answer": "Yes, Dubai Land Department charges a 4% transfer fee on the property’s purchase price."},
        {"question": "Q18: Are there property taxes in Dubai, Abu Dhabi, or Sharjah?", "answer": "No annual property taxes exist in the UAE, but buyers pay registration and service charges depending on the emirate."},
        {"question": "Q19: Can you get residency by buying property in the UAE?", "answer": "Yes, property investors can get 2–10-year residency visas depending on the value and location of the property."},
        {"question": "Q20: What are off-plan properties and are they safe to buy in the UAE?", "answer": "Off-plan properties are under construction or newly launched. Buying from approved developers (like Emaar, Aldar, Arada) ensures higher security."},
        {"question": "Q21: What is the difference between freehold and leasehold property in the UAE?", "answer": "Freehold gives the buyer full ownership of the property and land. Leasehold provides long-term rights (usually 10–99 years) to occupy or develop, without land ownership."},
        {"question": "Q22: Can foreigners buy freehold property in Dubai?", "answer": "Yes, foreigners can buy freehold property in approved areas such as Downtown, Dubai Marina, Palm Jumeirah, and Dubai Hills Estate."},
        {"question": "Q23: Is freehold ownership available for expats in Abu Dhabi and Sharjah?", "answer": "In Abu Dhabi, expats can buy freehold property only in designated investment zones. In Sharjah, expats can buy on a 100-year leasehold basis in approved areas like Aljada."},
        {"question": "Q24: What is Musataha and Usufruct in UAE real estate?", "answer": "Musataha grants the right to build and use land for up to 50 years, while Usufruct allows occupation or rental of property for up to 99 years."},
        {"question": "Q25: Can property ownership in the UAE lead to long-term residency?", "answer": "Yes. A minimum investment of AED 2 million in real estate can qualify the buyer for a 10-year UAE Golden Visa."},
        {"question": "Q26: What types of properties are available in Dubai?", "answer": "Dubai offers apartments, villas, townhouses, penthouses, duplexes, hotel apartments, and commercial properties."},
        {"question": "Q27: What are the popular property types in Abu Dhabi?", "answer": "In Abu Dhabi, common types include waterfront apartments, golf course villas, serviced apartments, and townhouses."},
        {"question": "Q28: What property types can expats buy in Sharjah?", "answer": "Expats can buy apartments, villas, and townhouses on long-term leasehold in selected developments like Aljada and Tilal City."},
        {"question": "Q29: Which is better: apartment or villa for investment in Dubai?", "answer": "Apartments offer higher rental yields in urban areas like Business Bay or Dubai Marina. Villas appreciate more in value over time, especially in areas like Palm Jumeirah or Dubai Hills."},
        {"question": "Q30: Can you buy commercial property in Dubai as a foreigner?", "answer": "Yes, foreigners can purchase commercial units in freehold zones such as Business Bay, JLT, and Downtown Dubai."},
        {"question": "Q31: What are the best communities for families in Dubai?", "answer": "Top choices include Dubai Hills Estate, Arabian Ranches, Jumeirah Golf Estates, and Mirdif for their schools, parks, and family amenities."},
        {"question": "Q32: Which Dubai communities offer the best rental returns?", "answer": "JVC, Arjan, International City, and Dubai Silicon Oasis are known for higher rental yields, especially in studio and 1-bedroom apartments."},
        {"question": "Q33: What is the best area for beachfront property in Dubai?", "answer": "Palm Jumeirah, JBR, Emaar Beachfront, and La Mer offer beachfront apartments and villas."},
        {"question": "Q34: Which Dubai areas are freehold for foreigners?", "answer": "Foreigners can buy in freehold zones such as Downtown Dubai, Dubai Marina, Dubai Creek Harbour, JVC, and Emirates Hills."},
        {"question": "Q35: Are there gated communities in Dubai?", "answer": "Yes, popular gated communities include Arabian Ranches, The Springs, The Meadows, DAMAC Hills, and Tilal Al Ghaf."},
        {"question": "Q36: What are the top investment zones in Abu Dhabi?", "answer": "Top zones include Al Reem Island, Saadiyat Island, Yas Island, Al Raha Beach, and Al Maryah Island."},
        {"question": "Q37: Is Saadiyat Island a freehold area?", "answer": "Yes, Saadiyat Island is a freehold investment zone where expats can buy villas and apartments."},
        {"question": "Q38: What makes Yas Island popular for investors?", "answer": "It offers entertainment (Ferrari World, Yas Mall), strong rental demand, and freehold villas and apartments from Aldar."},
        {"question": "Q39: Can you buy luxury property in Abu Dhabi?", "answer": "Yes. High-end villas and apartments are available in Al Saadiyat Island, Al Raha Beach, and Mamsha Al Saadiyat."},
        {"question": "Q40: What is the best area for waterfront living in Abu Dhabi?", "answer": "Al Raha Beach, Saadiyat Island, and Al Reem Island offer prime waterfront apartments."},
        {"question": "Q41: What are the best residential towers in Downtown Dubai?", "answer": "Popular towers include Burj Vista, The Address Downtown, Act One & Two, and Burj Royale."},
        {"question": "Q42: Which towers in Dubai Marina have the highest rental demand?", "answer": "Princess Tower, Marina Gate, Cayan Tower, and Elite Residence are highly sought-after by tenants and investors."},
        {"question": "Q43: What are the tallest residential towers in Dubai?", "answer": "Princess Tower, 23 Marina, Elite Residence, and The Torch are among the tallest."},
        {"question": "Q44: Which towers in Abu Dhabi are most popular for expats?", "answer": "Sky Tower, Sun Tower, The Gate Towers (Al Reem), and The Bridges are well-known choices."},
        {"question": "Q45: Are there any branded residences in towers?", "answer": "Yes. Dubai offers branded towers like Armani Residences (Burj Khalifa), One Za'abeel (One&Only), and Emaar Address Residences."},
        {"question": "Q46: Is there property tax in Dubai or Abu Dhabi?", "answer": "No annual property tax, but buyers pay a 4% transfer fee in Dubai and 2% in Abu Dhabi."},
        {"question": "Q47: Do I need a residency visa to own property in the UAE?", "answer": "No, but owning property worth AED 750K or more can qualify you for a 2-year visa; AED 2M or more qualifies for a 10-year Golden Visa."},
        {"question": "Q48: Can I get a mortgage as a non-resident?", "answer": "Yes, UAE banks offer mortgage options up to 50–70% for non-resident buyers with clear documentation."},
        {"question": "Q49: Is buying off-plan property safe in the UAE?", "answer": "Yes, if purchased from registered developers like Emaar, Aldar, or DAMAC and registered with the relevant land department."},
        {"question": "Q50: What happens to property after 99-year leasehold ends?", "answer": "Ownership reverts to the government or master developer unless extended or renewed under new terms."},
        {"question": "Q51: What zones allow expats to buy property in Sharjah?", "answer": "Aljada, Tilal City, Maryam Island, and Sharjah Waterfront City permit long-term leasehold for expats."},
        {"question": "Q52: Is Aljada a good place to invest in Sharjah?", "answer": "Yes, Aljada is a mixed-use community with rising demand, modern infrastructure, and strong developer backing (Arada)."},
        {"question": "Q53: What’s the lease period for properties in Sharjah for foreigners?", "answer": "Usually 100-year leaseholds for expats (non-GCC nationals) in permitted developments."},
        {"question": "Q54: Are service charges high in Sharjah properties?", "answer": "Service charges in Sharjah are generally lower than in Dubai, but vary based on project and location."},
        {"question": "Q55: Can expats own villas in Sharjah?", "answer": "Yes, in selected developments like Tilal City and Al Zahia on a long-term leasehold basis."},
        {"question": "Q56: What is the difference between off-plan and ready property in Dubai?", "answer": "Off-plan properties are under construction or newly launched, often with flexible payment plans. Ready properties are fully constructed and available for immediate handover."},
        {"question": "Q57: Is it safe to buy off-plan property in the UAE?", "answer": "Yes, if purchased from government-approved developers and registered with the Dubai Land Department (DLD) or Abu Dhabi Municipality."},
        {"question": "Q58: What are common off-plan payment plans in Dubai?", "answer": "Typical plans include 10% booking + 60/40 or 80/20 (construction/post-handover), depending on the developer."},
        {"question": "Q59: What is a post-handover payment plan?", "answer": "Buyers pay a portion of the property cost during construction and the remaining after receiving the keys over 1–5 years."},
        {"question": "Q60: How long does it take to receive handover after project completion?", "answer": "Usually within 1–3 months after completion, once final inspections and DLD registration are cleared."},
        {"question": "Q61: What is the average ROI for property investment in Dubai?", "answer": "ROI ranges from 6–9% in high-demand areas like JVC, Dubai Marina, and Business Bay for apartments."},
        {"question": "Q62: Which areas in Dubai have the highest rental yields?", "answer": "JVC, Arjan, IMPZ, and Dubai Sports City top the list for affordable apartment rentals with strong yields."},
        {"question": "Q63: Is buying a villa or apartment better for long-term capital gain in the UAE?", "answer": "Villas in prime locations (Palm Jumeirah, Emirates Hills) typically see higher capital appreciation over time."},
        {"question": "Q64: How do I calculate ROI on a UAE property investment?", "answer": "ROI = (Annual Rental Income – Service Charges) ÷ Property Purchase Price × 100%."},
        {"question": "Q65: Which emirate offers the best value for real estate investors?", "answer": "Dubai offers high liquidity and strong yield, while Abu Dhabi provides stability. Sharjah offers lower entry prices with long-term growth potential."},
        {"question": "Q66: How do I verify if a developer is registered in Dubai?", "answer": "Check via the DLD or RERA online portal using the project name or developer's trade license."},
        {"question": "Q67: Is it mandatory to register a property purchase with the DLD?", "answer": "Yes, all property transactions in Dubai must be registered with the Dubai Land Department (DLD)."},
        {"question": "Q68: What is the Oqood certificate in Dubai real estate?", "answer": "Oqood is the interim property registration issued by DLD for off-plan properties before final title deed issuance."},
        {"question": "Q69: How do I transfer property ownership in the UAE?", "answer": "You must visit the Land Department with all documents, pay the transfer fee (typically 4%), and complete registration."},
        {"question": "Q70: What is the escrow account law for off-plan properties?", "answer": "Developers must deposit buyer payments into an escrow account monitored by RERA to protect investor funds."},
        {"question": "Q71: Which is better: Downtown Dubai or Business Bay for investment?", "answer": "Downtown offers luxury and Burj Khalifa views with higher entry prices, while Business Bay offers better rental yields and more affordable options."},
        {"question": "Q72: Dubai Hills Estate vs. Arabian Ranches – which is better for families?", "answer": "Both are family-friendly. Dubai Hills has modern infrastructure and central location; Arabian Ranches offers more greenery and traditional community vibes."},
        {"question": "Q73: Is Palm Jumeirah a good place for short-term rental investments?", "answer": "Yes, Palm Jumeirah has strong demand for holiday homes and premium short-stay rentals, especially on Airbnb and Booking.com."},
        {"question": "Q74: JVC vs. Arjan – which area is better for budget investors?", "answer": "Both offer affordability, but JVC has better infrastructure, existing population, and higher rental demand."},
        {"question": "Q75: Al Reem Island vs. Saadiyat Island – which is better for lifestyle?", "answer": "Saadiyat Island offers luxury beachfront living and museums; Al Reem is more urban, with high-rise convenience."},
        {"question": "Q76: What are annual service charges in Dubai?", "answer": "They vary by property type and location, usually between AED 10–30 per sq. ft. per year."},
        {"question": "Q77: Who pays the service charges: owner or tenant?", "answer": "Owners pay annual service fees. Tenants may pay cooling (chiller) charges and DEWA separately."},
        {"question": "Q78: What is a chiller-free apartment in Dubai?", "answer": "It means cooling (air-conditioning) charges are included in the rent, reducing tenant utility expenses."},
        {"question": "Q79: How do I estimate maintenance costs for villas in Dubai?", "answer": "Villa maintenance can range from AED 10,000 to AED 30,000+ annually, depending on size and landscaping."},
        {"question": "Q80: Are service charges lower in Sharjah or Abu Dhabi compared to Dubai?", "answer": "Generally, yes. Service charges in Sharjah and Abu Dhabi are lower due to more government-regulated fees and simpler infrastructure."},
        {"question": "Q81: Is Airbnb legal in Dubai?", "answer": "Yes, owners must register with the Department of Economy and Tourism and get a holiday home license."},
        {"question": "Q82: Which Dubai areas are best for Airbnb investments?", "answer": "Downtown, Marina, JBR, Palm Jumeirah, and Emaar Beachfront are popular for short-term stays."},
        {"question": "Q83: Can I rent out my villa as a holiday home in Dubai?", "answer": "Yes, with proper licensing. Villas in Palm Jumeirah, Jumeirah Islands, and Meadows are in demand for luxury stays."},
        {"question": "Q84: Are there restrictions on short-term rentals in Abu Dhabi?", "answer": "Yes, operators must obtain a holiday home permit from the Department of Culture and Tourism (DCT)."},
        {"question": "Q85: How do I get a holiday home license in Dubai?", "answer": "Apply via the Dubai Tourism website, submit documents, pay fees, and comply with property standards."},
        {"question": "Q86: Is now a good time to buy property in Dubai?", "answer": "Yes. The market shows stable growth post-Expo 2020 and post-pandemic with rising demand, especially in off-plan and luxury segments."},
        {"question": "Q87: What is driving property prices in Dubai?", "answer": "Visa reforms, Golden Visa demand, business relocation, tax benefits, and high-end tourism are key drivers."},
        {"question": "Q88: Will Dubai property prices drop in 2025?", "answer": "Current forecasts suggest moderate growth with no major drop expected, especially in prime and freehold areas."},
        {"question": "Q89: Which areas in the UAE are expected to grow the most by 2026?", "answer": "Dubai South, Emaar South, Tilal Al Ghaf, and Saadiyat Island are among the top growth zones."},
        {"question": "Q90: Are property prices rising in Sharjah and Abu Dhabi too?", "answer": "Yes, with Sharjah seeing more demand due to affordability and Abu Dhabi’s continued development of Yas and Saadiyat Islands."},
        {"question": "Q91: What are off-plan properties in Dubai?", "answer": "Off-plan properties are those that are purchased before construction is completed — either at launch or during the build phase."},
        {"question": "Q92: Why do investors buy off-plan properties in Dubai?", "answer": "Off-plan properties offer lower entry prices, flexible payment plans, capital appreciation, and post-handover payment options."},
        {"question": "Q93: What are the risks of buying off-plan property in the UAE?", "answer": "Risks include construction delays, market fluctuations, or developer non-performance. These can be mitigated by buying from RERA-approved developers with escrow protection."},
        {"question": "Q94: How do I check if an off-plan project is registered with the Dubai Land Department (DLD)?", "answer": "You can verify project registration via the DLD REST app or through the DLD’s official website."},
        {"question": "Q95: What is the minimum down payment for off-plan projects in Dubai?", "answer": "Most projects start from a 5% to 20% booking deposit depending on the developer."},
        {"question": "Q96: Which are the best off-plan communities in Dubai in 2025?", "answer": "Top communities include Dubai Creek Harbour, Emaar South, Palm Jebel Ali, Tilal Al Ghaf, and The Oasis by Emaar."},
        {"question": "Q97: Can I sell an off-plan property in Dubai before completion?", "answer": "Yes, known as “assignment sales,” but you must meet the minimum payment threshold (often 40–60%) and get developer and DLD approval."},
        {"question": "Q98: How long does it take to hand over an off-plan property in Dubai?", "answer": "Most projects have a 2 to 4-year timeline, depending on the scale of development."},
        {"question": "Q99: Are off-plan properties in Dubai eligible for a Golden Visa?", "answer": "Yes, if the purchase value is AED 2 million or more and registered with DLD."},
        {"question": "Q100: Can off-plan property buyers apply for a mortgage in Dubai?", "answer": "Yes, selected banks offer mortgage approvals for off-plan purchases with approved developers, usually covering 50–75%."},
        {"question": "Q101: Are off-plan projects available for expat ownership in Abu Dhabi?", "answer": "Yes, in designated investment zones like Yas Island, Saadiyat Island, Al Reem, and Al Raha Beach."},
        {"question": "Q102: Who are the top off-plan developers in Abu Dhabi?", "answer": "Major developers include Aldar Properties, Bloom Holding, IMKAN, Reportage, and Q Properties."},
        {"question": "Q103: What are the best upcoming off-plan communities in Abu Dhabi?", "answer": "Reem Hills, Yas Acres, Saadiyat Lagoons, and The Sustainable City – Yas Island are among the most in-demand in 2025."},
        {"question": "Q104: Is there escrow protection for off-plan property buyers in Abu Dhabi?", "answer": "Yes. All off-plan projects must be registered, and buyer payments must go into escrow accounts regulated by the Department of Municipalities and Transport (DMT)."},
        {"question": "Q105: Can I resell off-plan property before completion in Abu Dhabi?", "answer": "Yes, but subject to developer approval and payment milestones."},
        {"question": "Q106: Can foreigners buy off-plan property in Sharjah?", "answer": "Yes, on a 100-year leasehold basis in approved communities like Aljada, Maryam Island, and Tilal City."},
        {"question": "Q107: Who are the main developers offering off-plan in Sharjah?", "answer": "Top names include Arada, Eagle Hills, Alef Group, and Tilal Properties."},
        {"question": "Q108: What is the average price for off-plan apartments in Sharjah?", "answer": "Studios typically start from AED 350,000; 1-bed from AED 450,000 depending on the location and project."},
        {"question": "Q109: Are there payment plans for Sharjah off-plan projects?", "answer": "Yes, flexible payment plans are available — typically 10/90 or 30/70 during and post construction."},
        {"question": "Q110: Do Sharjah off-plan properties come with a completion guarantee?", "answer": "Most projects are developer-backed but always verify the escrow and construction guarantee terms before purchasing."},
        {"question": "Q111: Are off-plan investments more profitable than ready properties?", "answer": "Off-plan offers higher capital appreciation potential; however, ready properties give immediate rental income. Choice depends on investment strategy."},
        {"question": "Q112: What happens if a Dubai off-plan project gets delayed?", "answer": "Buyers are protected under DLD laws. You may be entitled to compensation or refunds if delays breach contract terms."},
        {"question": "Q113: Can I cancel my off-plan purchase in Dubai?", "answer": "Yes, subject to RERA cancellation policy and contract terms. However, some portion of the paid amount may be deducted."},
        {"question": "Q114: What is the cost of transferring an off-plan unit to another buyer?", "answer": "Transfer fees range from AED 3,000 to AED 10,000, plus 4% DLD transfer fees (if selling before handover)."},
        {"question": "Q115: How do I check the construction status of an off-plan property?", "answer": "Use the DLD REST App or developer’s construction progress tracker (photos, timelines, RERA reports)."},
        {"question": "Q116: What documents should I receive when buying off-plan in Dubai?", "answer": "You should receive the SPA (Sales & Purchase Agreement), Oqood certificate, payment plan, and escrow account details."},
        {"question": "Q117: What is the role of the escrow account in off-plan deals?", "answer": "It safeguards buyer payments and ensures funds are only released to developers upon achieving construction milestones."},
        {"question": "Q118: Who pays the DLD registration fee for off-plan property?", "answer": "Buyers pay 4% of the property value, unless there’s a promotion where the developer pays partially or fully."},
        {"question": "Q119: Can I rent an off-plan property immediately after handover?", "answer": "Yes. Once the final handover is complete and the title deed is issued, the unit can be leased."},
        {"question": "Q120: What is the title deed issuance process after off-plan handover?", "answer": "Submit the handover certificate and clearance documents to DLD or the respective emirate’s land department for title deed issuance."},
        {"question": "Q121: What does freehold property mean in the UAE?", "answer": "Freehold means the buyer owns the property and the land it stands on, with full rights to sell, lease, or inherit it."},
        {"question": "Q122: Can foreigners buy freehold property in Dubai?", "answer": "Yes. Non-UAE nationals can buy freehold property in designated areas approved by the Dubai Land Department."},
        {"question": "Q123: What is the difference between freehold and leasehold in Dubai?", "answer": "Freehold gives lifetime ownership, while leasehold offers long-term lease rights (e.g. 99 years), often with some restrictions."},
        {"question": "Q124: Do freehold properties in the UAE come with residency visas?", "answer": "Yes. Properties worth AED 2M+ are eligible for a 10-year Golden Visa in Dubai and Abu Dhabi."},
        {"question": "Q125: Are there annual service charges for freehold property owners?", "answer": "Yes. Owners pay yearly service and maintenance fees, typically ranging between AED 10–30 per sq.ft."},
        {"question": "Q126: Is Downtown Dubai freehold?", "answer": "Yes. Foreigners can buy apartments in Burj Khalifa, Opera District, and surrounding towers."},
        {"question": "Q127: What is the average price per sq.ft in Downtown Dubai (2025)?", "answer": "Around AED 2,500 to AED 3,800 per sq.ft, depending on the tower and view."},
        {"question": "Q128: Can expats own property in Dubai Marina?", "answer": "Yes, it's one of the most popular freehold zones for high-rise apartments."},
        {"question": "Q129: Is Dubai Marina good for short-term rentals?", "answer": "Absolutely. It’s a hotspot for Airbnb and holiday homes due to its waterfront, nightlife, and walkability."},
        {"question": "Q130: Is Palm Jumeirah freehold for foreigners?", "answer": "Yes. Villas, apartments, and hotel-serviced units are available for full ownership."},
        {"question": "Q131: What types of properties are available in Palm Jumeirah?", "answer": "Signature villas, garden homes, shoreline apartments, penthouses, and branded residences."},
        {"question": "Q132: Can non-UAE citizens buy property in Dubai Hills Estate?", "answer": "Yes. It’s a popular freehold zone by Emaar with apartments, townhouses, and villas."},
        {"question": "Q133: Is Dubai Hills good for family living?", "answer": "Yes. It offers parks, schools, a golf course, and a mega mall in a gated environment."},
        {"question": "Q134: Is Business Bay freehold for foreigners?", "answer": "Yes. All residential towers are available for expat buyers."},
        {"question": "Q135: How does Business Bay compare to Downtown Dubai in ROI?", "answer": "Business Bay offers better rental yields (6–8%) while Downtown offers more capital appreciation."},
        {"question": "Q136: Is JVC a freehold area?", "answer": "Yes. It’s known for affordable apartments and family-friendly villas."},
        {"question": "Q137: Is JVC a good investment location in 2025?", "answer": "Yes. High rental yields and low entry points make it attractive for first-time buyers."},
        {"question": "Q138: Can expats buy in Dubai Creek Harbour?", "answer": "Yes. It’s a freehold community by Emaar with waterfront towers and views of Dubai skyline."},
        {"question": "Q139: When will Creek Tower be completed?", "answer": "Expected around 2026. It aims to be one of the tallest landmarks globally."},
        {"question": "Q140: Is Dubai South a freehold zone?", "answer": "Yes. Expat buyers can own villas, townhouses, and apartments in Emaar South and The Pulse."},
        {"question": "Q141: What is The Oasis by Emaar?", "answer": "The Oasis is a luxury villa community launched by Emaar in 2024, focused on exclusivity, greenery, and waterfront living."},
        {"question": "Q142: Where is The Oasis located in Dubai?", "answer": "It’s located near Dubai-Al Ain Road, close to Dubailand and Dubai Hills Estate."},
        {"question": "Q143: What property types are available in The Oasis?", "answer": "Luxury 4 to 6-bedroom villas with contemporary Arabic and Mediterranean designs."},
        {"question": "Q144: Are there payment plans for The Oasis project?", "answer": "Yes. Typically 80/20 or 90/10 plans with flexible post-handover options."},
        {"question": "Q145: Can foreigners buy off-plan villas on Palm Jebel Ali?", "answer": "Yes. Palm Jebel Ali offers 100% freehold ownership to all nationalities."},
        {"question": "Q146: What is the expected completion date for Palm Jebel Ali villas?", "answer": "Phase 1 is expected to be completed by Q4 2027."},
        {"question": "Q147: What is the average price of villas in Palm Jebel Ali?", "answer": "Starting from AED 18 million for beach and coral villas."},
        {"question": "Q148: Is Palm Jebel Ali eligible for a Golden Visa?", "answer": "Yes. Any property over AED 2 million qualifies."},
        {"question": "Q149: What is Bayview by Address in Emaar Beachfront?", "answer": "It’s a branded residence tower by Address Hotels with full sea views and 5-star amenities."},
        {"question": "Q150: What are the unit types in Bayview?", "answer": "1–4 bedroom apartments and penthouses with private beach access."},
        {"question": "Q151: Is Bayview a good investment in 2025?", "answer": "Yes. High ROI potential due to beachfront location, Address branding, and tourism demand."},
        {"question": "Q152: What is Sobha Hartland II?", "answer": "A luxury community with crystal lagoons, new tower launches, and villas — expansion of the original Hartland."},
        {"question": "Q153: Are the lagoons real and swimmable?", "answer": "Yes. It features artificial crystal lagoons like District One."},
        {"question": "Q154: What is Fairway Villas 3 in Emaar South?", "answer": "The latest golf-course-facing villas within Emaar South, near Dubai World Central."},
        {"question": "Q155: Is Emaar South a good location to invest now?", "answer": "Yes. Proximity to Al Maktoum Airport and Expo City boosts future ROI."},
        {"question": "Q156: What is the Morocco Cluster in Damac Lagoons?", "answer": "Themed around Moroccan architecture with lush courtyards and lagoon-facing villas."},
        {"question": "Q157: Are post-handover payment plans available?", "answer": "Yes. Damac offers 1–3 year post-handover plans on selected units."},
        {"question": "Q158: What is Saadiyat Lagoons?", "answer": "An eco-conscious, luxury villa community launched by Aldar on Saadiyat Island."},
        {"question": "Q159: Can expats buy in Saadiyat Lagoons?", "answer": "Yes. It's a designated freehold zone open to all nationalities."},
        {"question": "Q160: What is Gardenia Bay in Yas Island?", "answer": "Waterfront apartments with smart technology and resort-style living by Aldar."},
        {"question": "Q161: What is the starting price of apartments in Gardenia Bay?", "answer": "Studios start from around AED 765,000 with flexible payment plans."},
        {"question": "Q162: What is new in Reem Hills Phase 2?", "answer": "Larger plot sizes, private pools, and upgraded modern villa designs."},
        {"question": "Q163: What’s the newest launch in Aljada Sharjah?", "answer": "Boulevard 5 and Nest 2 — offering smart studios, 1BR and 2BR apartments with retail below."},
        {"question": "Q164: Can expats buy in Aljada?", "answer": "Yes. 100-year leasehold ownership for foreigners with full rights to sell or rent."},
        {"question": "Q165: What’s new on Maryam Island in 2025?", "answer": "Sapphire Residences, with full sea-view apartments and access to beach promenade."},
        {"question": "Q166: Is Maryam Island a good location for short-term rental?", "answer": "Yes. Its central waterfront location and urban design attract tourists and professionals alike."},
        {"question": "Q167: Are new off-plan projects in Dubai safe to invest in?", "answer": "Yes, if launched by top developers like Emaar, Nakheel, Aldar, and Arada, and registered with escrow accounts."},
        {"question": "Q168: Do new off-plan projects come with DLD waivers?", "answer": "Many developers offer limited-time 4% DLD fee waivers as launch promotions."},
        {"question": "Q169: How long are new off-plan project handover timelines?", "answer": "Most new projects launched in 2024–2025 aim for 2.5 to 4-year delivery."},
        {"question": "Q170: Which new off-plan communities offer post-handover payment plans?", "answer": "Damac Lagoons, Emaar South, Sobha Hartland II, and Gardenia Bay all offer post-handover options."},
        {"question": "Q171: Who are the most trusted developers in Dubai?", "answer": "The most reputable developers include Emaar, Nakheel, Meraas, Dubai Properties, Sobha, and Damac."},
        {"question": "Q172: What guarantees do UAE developers offer for off-plan projects?", "answer": "Most offer 1-year snagging warranties and 10-year structural warranties, in line with RERA regulations."},
        {"question": "Q173: How can I verify if a UAE developer is approved?", "answer": "Use the Dubai REST App or visit the official DLD or RERA portal to confirm developer and project status."},
        {"question": "Q174: Do UAE developers offer post-handover payment plans?", "answer": "Yes, especially for off-plan launches. Many offer 60/40 or 70/30 with up to 3-year post-handover terms."},
        {"question": "Q175: Who is Emaar Properties?", "answer": "Emaar is Dubai’s largest master developer, known for Downtown Dubai, Burj Khalifa, Dubai Hills, and Emaar Beachfront."},
        {"question": "Q176: What makes Emaar popular among investors?", "answer": "Emaar is known for timely handovers, strong resale value, high-quality finishing, and access to prime locations."},
        {"question": "Q177: What are some new projects by Emaar in 2025?", "answer": "The Oasis, Bayview, Palmiera Villas in The Oasis, and new towers in Creek Harbour and Dubai Hills."},
        {"question": "Q178: Does Emaar offer payment plans?", "answer": "Yes. Typical off-plan plans are 80/20 or 90/10 with flexible post-handover options."},
        {"question": "Q179: Is Damac a reliable real estate developer?", "answer": "Yes. Damac is known for large-scale themed communities like Damac Hills, Damac Lagoons, and branded towers."},
        {"question": "Q180: What is Damac Lagoons?", "answer": "A themed villa community with artificial lagoons, clustered by countries like Morocco, Venice, and Ibiza."},
        {"question": "Q181: Does Damac offer branded residences?", "answer": "Yes. Damac has partnered with Cavalli, de GRISOGONO, and Versace for branded towers."},
        {"question": "Q182: Are Damac projects good for rental ROI?", "answer": "Yes. Areas like Damac Hills and Lagoons offer 6–8% ROI due to affordability and demand."},
        {"question": "Q183: What is Nakheel famous for?", "answer": "Nakheel is the developer behind iconic Palm Jumeirah, Palm Jebel Ali, and The World Islands."},
        {"question": "Q184: Is Palm Jebel Ali developed by Nakheel?", "answer": "Yes. It’s their relaunch of the massive artificial island with beachfront villas and signature fronds."},
        {"question": "Q185: What are Nakheel’s current projects?", "answer": "Palm Jebel Ali, Jebel Ali Village Villas, and developments in Dubai Islands."},
        {"question": "Q186: What makes Sobha Realty unique in Dubai?", "answer": "Sobha offers in-house construction and premium build quality, especially in Sobha Hartland and Hartland II."},
        {"question": "Q187: Is Sobha Hartland a freehold community?", "answer": "Yes. Located in MBR City with luxury waterfront apartments and villas."},
        {"question": "Q188: Do Sobha projects offer post-handover plans?", "answer": "Yes. Many projects offer 60/40 plans and high-end finishes."},
        {"question": "Q189: What are Meraas’ major Dubai developments?", "answer": "City Walk, Bluewaters Island, Jumeirah Bay Island, and Port de La Mer."},
        {"question": "Q190: Are Meraas projects considered luxury?", "answer": "Yes. Their portfolio is focused on premium urban beachfront and designer developments."},
        {"question": "Q191: Can foreigners buy property in Bluewaters or Port de La Mer?", "answer": "Yes. Meraas developments are 100% freehold and open to all nationalities."},
        {"question": "Q192: Who owns Dubai Properties?", "answer": "It’s part of Dubai Holding, the government-owned master developer."},
        {"question": "Q193: What are some key projects by DP?", "answer": "Jumeirah Beach Residence (JBR), Business Bay, Mudon, Villanova, and Dubai Wharf."},
        {"question": "Q194: Are DP communities good for families?", "answer": "Yes. Communities like Mudon and Villanova are gated and designed for family living."},
        {"question": "Q195: Is Ellington a good developer for boutique buyers?", "answer": "Yes. They focus on design-driven apartments in JVC, Downtown, and Palm Jumeirah."},
        {"question": "Q196: Are Ellington projects off-plan or ready?", "answer": "Primarily off-plan, but with fast delivery and award-winning architecture."},
        {"question": "Q197: Is Aldar a government-backed developer?", "answer": "Yes. It’s the largest master developer in Abu Dhabi, known for Yas Island, Saadiyat Island, and Al Ghadeer."},
        {"question": "Q198: Can expats buy in Aldar communities?", "answer": "Yes. Aldar offers freehold property to all nationalities in designated zones."},
        {"question": "Q199: What new projects has Aldar launched in 2025?", "answer": "Gardenia Bay, Saadiyat Lagoons Phase 2, and new launches on Reem and Jubail Islands."},
        {"question": "Q200: Are Aldar projects eligible for Golden Visas?", "answer": "Yes. AED 2M+ investments in Aldar properties make buyers eligible for 10-year visas."},
        {"question": "Q201: Who is Arada?", "answer": "Arada is Sharjah’s largest private developer, behind Aljada, Masaar, and Nest student housing."},
        {"question": "Q202: Can foreigners buy in Arada’s projects?", "answer": "Yes. Arada offers 100-year leasehold property ownership with full resale and rental rights."},
        {"question": "Q203: Is Aljada good for investment in 2025?", "answer": "Yes. It’s a master community with schools, hotels, malls, and residential blocks ideal for rental yield."},
        {"question": "Q204: What is Maryam Island by Eagle Hills?", "answer": "A waterfront community in Sharjah offering beachfront apartments and retail promenade."},
        {"question": "Q205: Is Eagle Hills a reputable developer?", "answer": "Yes. It’s based in Abu Dhabi and has delivered successful mixed-use projects across the UAE and MENA region."},
        {"question": "Q206: Which Dubai developer offers the best post-handover plans?", "answer": "Damac and Emaar often provide flexible 2–4 year post-handover payment plans."},
        {"question": "Q207: Which developer has the best ROI properties in 2025?", "answer": "Damac (affordable high-yield units), Emaar (long-term capital appreciation), and Sobha (high-end end-user demand)."},
        {"question": "Q208: Who builds the best quality luxury villas in Dubai?", "answer": "Sobha Realty and Emaar are top contenders in terms of construction and finishing quality."},
        {"question": "Q209: Are branded residences only built by certain developers?", "answer": "Yes. Damac (Cavalli, Versace), Emaar (Address, Vida), and Omniyat (Dorchester, One Palm) lead branded segments."},
        {"question": "Q210: Which developer is ideal for first-time investors?", "answer": "Dubai Properties, Ellington, and Arada offer lower price points with high rental demand."},
        {"question": "Q211: Can foreigners buy property in Dubai?", "answer": "Yes. Foreign nationals can buy property in designated freehold zones in Dubai with 100% ownership rights."},
        {"question": "Q212: Can expats buy property in Abu Dhabi?", "answer": "Yes. Expats can buy in freehold areas like Yas Island, Al Reem, Saadiyat Island, and Al Maryah Island."},
        {"question": "Q213: Is foreign property ownership allowed in Sharjah?", "answer": "Yes. Sharjah offers 100-year leasehold ownership to non-GCC nationals in projects like Aljada and Masaar."},
        {"question": "Q214: What is the difference between freehold and leasehold property?", "answer": "Freehold means you own the property and land forever. Leasehold is long-term (usually 99–100 years) without land ownership."},
        {"question": "Q215: Can I own property jointly with my spouse or partner?", "answer": "Yes. Joint ownership is legal, and both names can appear on the title deed."},
        {"question": "Q216: What is a title deed in Dubai?", "answer": "A legal document issued by the Dubai Land Department (DLD) proving full ownership of the property."},
        {"question": "Q217: What is an Oqood certificate in off-plan purchases?", "answer": "An Oqood is a DLD-issued pre-title registration certificate for off-plan properties until handover."},
        {"question": "Q218: How long does it take to receive a title deed after purchase?", "answer": "If it's a ready property, the title deed is issued within 1–2 weeks after full payment and DLD registration."},
        {"question": "Q219: What is RERA in Dubai?", "answer": "The Real Estate Regulatory Agency (RERA) regulates developers, brokers, and property transactions under DLD."},
        {"question": "Q220: How do I verify a project's legal status in Dubai?", "answer": "Use the Dubai REST app to check project approvals, escrow accounts, and developer licensing."},
        {"question": "Q221: Do all off-plan projects require an escrow account?", "answer": "Yes. All off-plan developments must have a DLD-monitored escrow account to protect buyers."},
        {"question": "Q222: Is it mandatory to use a registered real estate broker?", "answer": "Yes. Brokers must hold valid RERA cards and be licensed under the Dubai Economic Department."},
        {"question": "Q223: What is Form F in Dubai property transactions?", "answer": "Form F is the official sale and purchase agreement (SPA) used in every property transaction in Dubai."},
        {"question": "Q224: Is there property tax in the UAE?", "answer": "No. The UAE has no annual property tax, inheritance tax, or capital gains tax on real estate."},
        {"question": "Q225: What is the DLD transfer fee in Dubai?", "answer": "It’s 4% of the property value, plus AED 580 admin fee, payable during title deed registration."},
        {"question": "Q226: Can property owners in the UAE get a Golden Visa?", "answer": "Yes. Foreigners investing AED 2 million+ in UAE real estate can apply for a 10-year Golden Visa."},
        {"question": "Q227: Is VAT applicable to property purchases in Dubai?", "answer": "No VAT on residential property sales. But VAT may apply on commercial units or newly built properties within 3 years of handover."},
        {"question": "Q228: What happens to my UAE property if I die without a will?", "answer": "Sharia law may apply. Expats are encouraged to register a will at DIFC Wills Centre to ensure inheritance follows their wishes."},
        {"question": "Q229: Can I register a will for my Dubai property?", "answer": "Yes. DIFC or Abu Dhabi Judicial Department (ADJD) allows expats to register English-language wills for real estate."},
        {"question": "Q230: Can my children inherit my Dubai property?", "answer": "Yes, if a will is in place. Otherwise, local courts may follow Sharia distribution laws."},
        {"question": "Q231: What if a developer delays the off-plan project delivery?", "answer": "Buyers are protected via DLD escrow laws. In severe cases, RERA can cancel projects and refund from escrow."},
        {"question": "Q232: Can I cancel my off-plan purchase?", "answer": "Yes, under certain conditions (delays, breaches). However, a penalty may apply depending on the SPA."},
        {"question": "Q233: Are real estate disputes handled in court or arbitration?", "answer": "Disputes are typically handled via Dubai Rental Disputes Centre (RDC) or Property Courts, depending on the case type."},
        {"question": "Q234: Can I sell an off-plan property before handover?", "answer": "Yes. This is called an assignment sale — subject to developer approval and minimum payment thresholds (often 30–40%)."},
        {"question": "Q235: Can I get a mortgage for an off-plan property in Dubai?", "answer": "Yes. Some banks offer off-plan project financing — especially for Emaar, Nakheel, Aldar, etc., with 50–70% LTV."},
        {"question": "Q236: What documents are required to buy property as a non-resident?", "answer": "Valid passport, proof of funds, buyer declaration, and sometimes bank reference letter."},
        {"question": "Q237: Are all developers required to register their project?", "answer": "Yes. Developers must register with RERA and DLD and link each project to an escrow account."},
        {"question": "Q238: Can I get a refund if the developer fails to deliver the project?", "answer": "Yes. RERA may refund buyers from escrow accounts if a project is officially canceled."},
        {"question": "Q239: Are buyers protected when buying in newly launched projects?", "answer": "Yes. Projects must be DLD-approved, escrow-regulated, and supervised under Law No. 8 of 2007."},
        {"question": "Q240: What are the legal fees involved in a Dubai property deal?", "answer": "4% DLD transfer fee, AED 5,250 to 10,000 trustee office fee, AED 2,000–5,000 broker fee (if applicable), AED 1,000–2,000 for NOC or admin fees."},
        {"question": "Q241: Can someone else buy on my behalf with a Power of Attorney?", "answer": "Yes. A POA can be notarized in the UAE or legalized from abroad for real estate transactions."},
        {"question": "Q242: How long does a Dubai property transaction take?", "answer": "For ready property: 5–10 days. For off-plan: depends on developer, usually 1–2 weeks."},
        {"question": "Q243: What is the average price of property in Dubai in 2025?", "answer": "As of 2025, the average price is around AED 1,450–1,750 per sq.ft., depending on location, developer, and type."},
        {"question": "Q244: What is the current price trend in Dubai real estate?", "answer": "Prices are increasing moderately in prime areas and master communities, especially for villas and branded residences."},
        {"question": "Q245: Are property prices going up or down in Dubai?", "answer": "Prices are stable to rising in most communities due to population growth, investor demand, and Golden Visa incentives."},
        {"question": "Q246: What are the most expensive areas to buy property in Dubai?", "answer": "Palm Jumeirah: AED 3,500–7,000/sq.ft.; Downtown Dubai: AED 2,500–4,000/sq.ft.; Dubai Hills Estate – Golf Front: AED 2,000–3,000/sq.ft.; Jumeirah Bay Island: AED 6,000–8,000/sq.ft."},
        {"question": "Q247: What are the most affordable areas to buy property in Dubai?", "answer": "Dubai South (Emaar South, MAG): AED 750–1,000/sq.ft.; Jumeirah Village Circle (JVC): AED 950–1,250/sq.ft.; International City: AED 500–750/sq.ft."},
        {"question": "Q248: What is the price of an apartment in Dubai Marina?", "answer": "Prices range from AED 1,600 to 3,000/sq.ft., depending on tower, view, and floor."},
        {"question": "Q249: What is the average villa price in Dubai?", "answer": "3BR Townhouse: AED 1.8M – AED 2.6M; 4BR Villa: AED 2.8M – AED 5.5M; Luxury Signature Villas: AED 15M – AED 250M+"},
        {"question": "Q250: What is the cheapest community to buy a villa in Dubai?", "answer": "Dubailand (Villanova, Rukan): From AED 1.6M; DAMAC Lagoons: 3BR from AED 1.85M; Emaar South: Starting AED 1.75M."},
        {"question": "Q251: What are the prices of 10-bedroom mansions in Emirates Hills?", "answer": "From AED 80M up to AED 250M+, depending on plot size, golf view, and customization."},
        {"question": "Q252: Are off-plan properties cheaper than ready units?", "answer": "Yes. Off-plan units are often 10–20% cheaper and come with flexible payment plans."},
        {"question": "Q253: What is the starting price of new off-plan projects in Dubai?", "answer": "Studio: AED 500K – AED 800K; 1BR: AED 900K – AED 1.4M; 2BR: AED 1.5M – AED 2.2M; 3BR Villas: AED 1.9M – AED 3.5M."},
        {"question": "Q254: What is the price per sq.ft. for off-plan in Creek Harbour and Beachfront?", "answer": "Emaar Beachfront: AED 3,000–4,200/sq.ft.; Creek Harbour: AED 1,600–2,200/sq.ft."},
        {"question": "Q255: Are there off-plan properties under AED 1 million in 2025?", "answer": "Yes. Projects in JVC, Dubai South, Arjan, Dubailand, and Town Square offer studios and 1BRs under AED 1M."},
        {"question": "Q256: What is the average price of property in Abu Dhabi in 2025?", "answer": "Apartments: AED 1,100–1,500/sq.ft.; Villas: AED 1,200–2,500/sq.ft.; Premium Islands (Saadiyat, Yas): AED 2,800–4,500/sq.ft."},
        {"question": "Q257: What is the price of Aldar’s Saadiyat Lagoons Villas?", "answer": "4–5BR villas range from AED 6M to AED 12M+, depending on plot and lagoon view."},
        {"question": "Q258: Is property cheaper in Sharjah than in Dubai?", "answer": "Yes. Sharjah offers 60–70% lower prices, starting from AED 500–900/sq.ft. in leasehold communities."},
        {"question": "Q259: What are Arada’s project prices in Aljada or Masaar?", "answer": "Apartments in Aljada: From AED 450K; Villas in Masaar: From AED 1.7M; Townhouses: From AED 1.4M."},
        {"question": "Q260: Can investors get ROI in Sharjah like Dubai?", "answer": "Yes. Yields in new Sharjah communities range from 6–9%, especially in Arada or Eagle Hills developments."},
        {"question": "Q261: Are resale prices higher than off-plan prices in Dubai?", "answer": "In popular areas, off-plan properties see 15–35% value growth by handover."},
        {"question": "Q262: Which areas in Dubai are giving the best capital appreciation?", "answer": "Dubai Creek Harbour, Dubai Hills Estate, Palm Jebel Ali, Damac Lagoons (due to early-phase pricing)."},
        {"question": "Q263: What is the ROI in Dubai rental market in 2025?", "answer": "Apartments: 6% – 8.5%; Villas: 5% – 7%; Short-term rentals: 8% – 12% in tourist zones."},
        {"question": "Q264: Are Dubai property prices expected to rise further in 2025–26?", "answer": "Yes. Continued investor migration, limited villa supply, and mega launches like Palm Jebel Ali will drive demand and prices."},
        {"question": "Q265: What is the minimum down payment to buy property in Dubai?", "answer": "For residents with mortgage: 20%; For foreigners: 20–25% (plus DLD fees); Off-plan: as low as 5–10%."},
        {"question": "Q266: Are payment plans available for ready properties?", "answer": "Rarely. Developers like Damac, Sobha, or select sellers offer post-handover plans on limited inventory."},
        {"question": "Q267: What are the total closing costs to buy property in Dubai?", "answer": "4% DLD Transfer Fee, AED 5,250 Title Deed Fee, Broker Commission: 2% (if applicable), Trustee Fee: AED 4,200–5,250."}
    ]

    # Initialize Sentence Transformer for embeddings
    model = SentenceTransformer(EMBED_MODEL_ID)

    # Initialize Qdrant Client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Create collection if it doesn't exist
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Collection '{COLLECTION_NAME}' recreated.")
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' already exists or error creating: {e}. Skipping recreation.")

    # Prepare data for upsert
    points = []
    for i, faq in enumerate(FAQ_DATA):
        if not faq.get("question") or not faq.get("answer"):
            print(f"Skipping malformed FAQ at index {i}: {faq}")
            continue

        question_text = faq["question"]
        answer_text = faq["answer"]
        embedding = model.encode(question_text).tolist()  # Embed the question
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),  # Generate a unique ID for each point
                vector=embedding,
                payload={"question": question_text, "answer": answer_text}
            )
        )

    print(f"Prepared {len(points)} points for upsert. Expected: {len(FAQ_DATA)}")
    if len(points) != len(FAQ_DATA):
        print("WARNING: The number of prepared points does not match the number of FAQs in the dataset.")


    # Upsert points to Qdrant
    try:
        operation_info = client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        print(f"Upserted {len(points)} points: {operation_info}")
    except Exception as e:
        print(f"Error during upsert: {e}")

    print("Data ingestion complete.")

# Schedule the job to run every 24 hours
# schedule.every(24).hours.do(job)

# To run the job immediately once, and then according to the schedule
job()

# while True:
#     schedule.run_pending()
#     time.sleep(1)
