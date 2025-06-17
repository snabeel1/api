import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
COLLECTION_NAME = "FAQS_COLLECTION" 
EMBED_MODEL_ID = "all-MiniLM-L6-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# --- CATEGORIZED FAQ DATA ---
# This dictionary holds your FAQs, categorized by the topics you provided.
# The new APIL GPT category is added at the beginning.
CATEGORIZED_FAQ_DATA = {
    "APIL GPT Specific FAQs": [
        {"question": "How APIL GPT Works as a Game-Changer for Dubai Real Estate?", "answer": "APIL GPT redefines the Dubai real estate market by offering speed, precision, and 24/7 AI intelligence as an AI Property Assistant. It provides instant answers on off-plan projects, rental listings, and developer comparisons, making it Dubai’s #1 AI Chatbot for homebuyers, investors, and real estate agents."},
        {"question": "What is APIL GPT?", "answer": "APIL GPT is an AI chatbot for real estate that functions as a personal assistant, providing immediate answers to property-related questions anytime, anywhere."},
        {"question": "How does APIL GPT work?", "answer": "It uses advanced AI to understand questions posed in simple English and delivers meaningful, data-backed answers on real estate much faster than manual searches."},
        {"question": "Is APIL GPT only for homebuyers?", "answer": "No. It is designed for anyone needing quick, precise property information in Dubai, including buyers, tenants, investors, and real estate professionals."},
        {"question": "Can APIL GPT handle commercial real estate, too?", "answer": "Yes, it serves as an effective chatbot for commercial real estate, facilitating searches for office spaces, retail units, and investment-grade properties."},
        {"question": "What kind of questions can I ask APIL GPT?", "answer": "You can ask about a wide range of topics, including property rates, payment plans, return on investment (ROI), and project timelines. Examples include 'What are the best ROI properties in JVC?' or 'Show me villas under AED 2M?'."},
        {"question": "Is it better than a real estate agent?", "answer": "It serves as an AI real estate agent and a valuable addition to an agent's work, providing instant answers that aid in faster and more informed decision-making. It's not a substitute but a powerful assistant."},
        {"question": "Why is APIL GPT called a “game-changer”?", "answer": "It's a game-changer because it significantly saves time, reduces confusion, and enhances the accuracy of decisions by providing comprehensive real estate insights instantly."},
        {"question": "Is APIL GPT free?", "answer": "Yes. It is entirely free to use, requiring no login or hidden fees. You can access it by visiting chat.apilproperties.com and starting a chat."},
        {"question": "How is it useful for real estate agents?", "answer": "It's an ideal chatbot for real estate agents, enabling them to automate responses, generate leads, and manage multiple clients concurrently, particularly for off-plan and high-demand properties."},
        {"question": "What makes it different from other chatbots?", "answer": "Unlike other bots, APIL GPT is specifically trained for the Dubai real estate market, possessing extensive knowledge of local regions, prices, developers, and trends."},
        {"question": "Is APIL GPT available 24/7?", "answer": "Yes. It functions as an AI virtual assistant that is always online, providing immediate answers without any wait or follow-up required."},
        {"question": "Can it be used on mobile?", "answer": "Yes, APIL GPT is mobile-responsive and accessible via your phone's browser, making it convenient for on-the-go property searches."},
        {"question": "Can APIL GPT help me find off-plan projects?", "answer": "Yes, it specializes in off-plan listings and can filter by developer, area, handover date, and payment plan, making it highly valuable for investors."},
        {"question": "Does APIL GPT provide rental property options?", "answer": "Yes, it offers immediate listings for both long-term and short-term furnished apartments."},
        {"question": "How accurate is the data from APIL GPT?", "answer": "The data is highly accurate, sourced from current databases and verified listings, providing real-time information on price ranges, sizes, and availability."},
        {"question": "Can I compare two properties using APIL GPT?", "answer": "Yes. You can request comparisons between projects based on developer reputation, ROI, amenities, or location, facilitating intelligent decision-making."},
        {"question": "Is APIL GPT suitable for international investors?", "answer": "Yes! It was made for people from all over the world and answers questions about anything from who can get a visa to how overseas buyers can pay."},
        {"question": "Can it calculate expected ROI?", "answer": "Yes. As an AI real estate consultant, APIL GPT can show you how much money you can make renting out properties and how much they will go up in value over time in different Dubai neighbourhoods."},
        {"question": "Can I ask for properties near schools or hospitals?", "answer": "Yes. If you ask 'Show me 2-bedroom apartments near top schools in Dubai' or 'Homes near Dubai Healthcare City,' it will help."},
        {"question": "How does APIL GPT help during property launches?", "answer": "It gives you launch costs, unit sizes, and floor plans right away, which makes it the best way to swiftly find your way around new project announcements."},
        {"question": "Can APIL GPT recommend developers?", "answer": "Yes. You may acquire project information right away by asking for leading Dubai developers like Emaar, DAMAC, Sobha, or Ellington."},
        {"question": "Is APIL GPT suitable for luxury property searches?", "answer": "Yes, for sure. It's great for wealthy buyers who want penthouses, branded homes, or beachfront villas with high-end equipment."},
        {"question": "Can APIL GPT show trending communities in Dubai?", "answer": "Yes. If you ask for the places that are most looked for or that sell the fastest, APIL GPT will show you hot spots like Dubai Marina, JVC, or Business Bay."},
        {"question": "Is APIL GPT secure to use?", "answer": "Yes. It runs on encrypted platforms and doesn't collect any personal information, so your search is safe and confidential."},
        {"question": "Can it assist in legal or documentation queries?", "answer": "It can clarify common legal words like DLD registration, title deed, or NOC, but for more in-depth legal guidance, you should talk to a certified specialist."},
        {"question": "Can I ask about top developers?", "answer": "Yes. You can ask, 'Is this a trustworthy developer?' and APIL GPT will give you a quick review of their reputation and past projects."},
    ],
    "Property Ownership": [
        {"question": "Can foreigners buy property in Dubai?", "answer": "Yes, foreigners can buy freehold properties in designated areas of Dubai like Downtown Dubai, Dubai Marina, Palm Jumeirah, and Jumeirah Village Circle (JVC)."},
        {"question": "What is the difference between freehold and leasehold in Dubai?", "answer": "Freehold allows full ownership of the property and land, while leasehold grants rights for 10 to 99 years without land ownership."},
        {"question": "Can you own property in Downtown Dubai as a non-resident?", "answer": "Yes, Downtown Dubai is a designated freehold area where non-residents can have 100% ownership."},
        {"question": "Can expats buy property in Abu Dhabi?", "answer": "Yes, expats can buy property in designated investment zones in Abu Dhabi, such as Yas Island, Saadiyat Island, and Al Reem Island."},
        {"question": "What types of ownership are available for expats in Abu Dhabi?", "answer": "In Abu Dhabi, expats can own freehold properties in investment zones, or opt for Musataha (long-term development right) or Usufruct (long-term right to use)."},
        {"question": "Can foreigners buy property in Sharjah?", "answer": "Foreigners can buy property in Sharjah under a 100-year leasehold agreement in designated development areas like Aljada and Masaar."},
        {"question": "What is the difference between property ownership in Sharjah and Dubai?", "answer": "Dubai offers widespread freehold ownership to foreigners, while Sharjah primarily offers 100-year leasehold for non-GCC nationals."},
        {"question": "What types of properties can you buy in Sharjah?", "answer": "In Sharjah, you can buy apartments, townhouses, and villas in master-planned leasehold communities, often within mixed-use developments."},
        {"question": "What is the difference between freehold and leasehold property in the UAE?", "answer": "Freehold grants full ownership of the land and property in perpetuity, while leasehold grants usage rights for a specific period (e.g., 99 years) without owning the land."},
        {"question": "Can foreigners buy freehold property in Dubai?", "answer": "Yes, foreign nationals can purchase freehold property with 100% ownership in designated areas (freehold zones) across Dubai."},
        {"question": "Is freehold ownership available for expats in Abu Dhabi and Sharjah?", "answer": "Freehold is available for expats in specific investment zones in Abu Dhabi. Sharjah primarily offers 100-year leasehold for non-GCC expats."},
        {"question": "What is Musataha and Usufruct in UAE real estate?", "answer": "Musataha is a right to build on or develop land for up to 50 years, renewable. Usufruct is a right to use property for up to 99 years. Both are common in Abu Dhabi for expat ownership outside freehold zones."},
        {"question": "What types of properties are available in Dubai?", "answer": "Dubai offers a wide range including apartments (studios to penthouses), villas, townhouses), commercial offices, retail units, and plots of land."},
        {"question": "What property types can you buy in Abu Dhabi?", "answer": "Abu Dhabi features apartments, villas, townhouses, and commercial properties, particularly in its investment zones."},
        {"question": "What property types can expats buy in Sharjah?", "answer": "Expats can purchase apartments, townhouses, and villas in designated leasehold areas of Sharjah, primarily within new master-planned communities."},
        {"question": "Can you buy commercial property in Dubai as a foreigner?", "answer": "Yes, foreigners can buy commercial properties (offices, retail units) in freehold areas across Dubai, often in business districts like Business Bay or Downtown."},
        {"question": "Do I need a residency visa to own property in the UAE?", "answer": "No, you don't need a residency visa to purchase property in the UAE. However, owning property of a certain value can make you eligible for a Golden Visa."},
        {"question": "What happens to property after 99-year leasehold ends?", "answer": "Upon expiry of a 99-year leasehold, ownership reverts to the landowner, unless the lease is renewed or an extension option is exercised as per the original agreement."},
        {"question": "What’s the lease period for properties in Sharjah for foreigners?", "answer": "The standard leasehold period for foreigners in designated Sharjah developments is 100 years, providing long-term ownership rights."},
        {"question": "Can expats own villas in Sharjah?", "answer": "Yes, expats can own villas in designated leasehold communities in Sharjah, such as Aljada and Masaar."},
        {"question": "What does freehold property mean in the UAE?", "answer": "Freehold property in the UAE grants the owner full, perpetual ownership of both the land and the building constructed on it, without time limit."},
        {"question": "Can foreigners buy freehold property in Dubai?", "answer": "Yes, foreign nationals can purchase freehold property with 100% ownership in specific designated areas across Dubai."},
        {"question": "What is the difference between freehold and leasehold in Dubai?", "answer": "Freehold grants complete and indefinite ownership of the property and land, while leasehold grants the right to use and occupy a property for a fixed period, usually 10 to 99 years, without owning the land."},
        {"question": "Is Palm Jumeirah freehold for foreigners?", "answer": "Yes, Palm Jumeirah is a designated freehold area in Dubai, allowing foreigners to buy properties with 100% ownership."},
        {"question": "What types of properties are available in Palm Jumeirah?", "answer": "Palm Jumeirah offers luxury apartments, beachfront villas (often with private pools), and hotel residences."},
        {"question": "Can foreigners buy off-plan villas on Palm Jebel Ali?", "answer": "Yes, Palm Jebel Ali is a freehold development open to all nationalities, including foreigners, for off-plan villa purchases."},
        {"question": "Can expats buy in Aljada?", "answer": "Yes. 100-year leasehold ownership for foreigners with full rights to sell or rent."},
        {"question": "Can foreigners buy property in Dubai?", "answer": "Yes. Foreign nationals can buy property in designated freehold zones in Dubai with 100% ownership rights."},
        {"question": "Can expats buy property in Abu Dhabi?", "answer": "Yes. Expats can buy in freehold areas like Yas Island, Al Reem, Saadiyat Island, and Al Maryah Island."},
        {"question": "Is foreign property ownership allowed in Sharjah?", "answer": "Yes. Sharjah offers 100-year leasehold ownership to non-GCC nationals in projects like Aljada and Masaar."},
        {"question": "What is the difference between freehold and leasehold property?", "answer": "Freehold means you own the property and land forever. Leasehold is long-term (usually 99–100 years) without land ownership."},
        {"question": "Can I own property jointly with my spouse or partner?", "answer": "Yes. Joint ownership is legal, and both names can appear on the title deed."},
        {"question": "What is a title deed in Dubai?", "answer": "A legal document issued by the Dubai Land Department (DLD) proving full ownership of the property."},
        {"question": "What is an Oqood certificate in off-plan purchases?", "answer": "An Oqood is a DLD-issued pre-title registration certificate for off-plan properties until handover."},
        {"question": "How long does it take to receive a title deed after purchase?", "answer": "If it's a ready property, the title deed is issued within 1–2 weeks after full payment and DLD registration."},
        {"question": "What is Form F in Dubai property transactions?", "answer": "Form F is the official sale and purchase agreement (SPA) used in every property transaction in Dubai."},
        {"question": "Can someone else buy on my behalf with a Power of Attorney?", "answer": "Yes. A POA can be notarized in the UAE or legalized from abroad for real estate transactions."},
    ],
    "Community Zones & Freehold Areas": [
        {"question": "What are the top freehold communities in Dubai for expats?", "answer": "Popular freehold communities for expats in Dubai include Dubai Marina, Downtown Dubai, Palm Jumeirah, Jumeirah Village Circle (JVC), and Dubai Hills Estate."},
        {"question": "Which are the best towers in Dubai Marina for investment?", "answer": "Popular towers for investment in Dubai Marina include Princess Tower, Cayan Tower, Marina Gate, and Emaar's Marina Promenade buildings, known for views and amenities."},
        {"question": "Which are the most popular communities in Abu Dhabi for property investment?", "answer": "Top communities for property investment in Abu Dhabi include Yas Island, Saadiyat Island, Al Reem Island, and Al Maryah Island."},
        {"question": "What are the best residential towers in Al Reem Island?", "answer": "Notable residential towers in Al Reem Island include Gate Towers, Sky Tower, Sun Tower, and The Bridges, offering modern apartments and amenities."},
        {"question": "What are the top communities to buy property in Sharjah?", "answer": "Leading communities for property purchase in Sharjah are Aljada, Masaar, and Maryam Island, all offering master-planned developments with amenities."},
        {"question": "Are there free zones in Sharjah where expats can invest in property?", "answer": "While not designated 'free zones' for property like in Dubai, Sharjah has specific development areas like Aljada and Maryam Island where foreigners can purchase property under a 100-year leasehold."},
        {"question": "What are the best communities for families in Dubai?", "answer": "Top family-friendly communities in Dubai include Dubai Hills Estate, Arabian Ranches, Mudon, Villanova, and Damac Hills, known for villas, green spaces, and amenities."},
        {"question": "Which Dubai areas are freehold for foreigners?", "answer": "Major freehold areas in Dubai for foreign ownership include Downtown Dubai, Dubai Marina, Palm Jumeirah, Jumeirah Lakes Towers (JLT), Jumeirah Village Circle (JVC), Business Bay, and Dubai Hills Estate."},
        {"question": "Are there gated communities in Dubai?", "answer": "Yes, many popular residential areas in Dubai are gated communities, offering enhanced security, privacy, and community facilities like parks and pools (e.g., Arabian Ranches, Emirates Hills)."},
        {"question": "What are the top investment zones in Abu Dhabi?", "answer": "The primary investment zones in Abu Dhabi for foreign ownership are Yas Island, Saadiyat Island, Al Reem Island, and Al Maryah Island, known for mixed-use developments."},
        {"question": "Is Saadiyat Island a freehold area?", "answer": "Yes, Saadiyat Island is a designated freehold investment zone in Abu Dhabi, allowing foreigners to purchase properties with full ownership."},
        {"question": "What makes Yas Island popular for investors?", "answer": "Yas Island is popular due to its entertainment attractions (Ferrari World, Warner Bros. World), F1 circuit, golf courses, and ongoing development of residential and commercial properties, offering high rental demand."},
        {"question": "What is the best area for waterfront living in Abu Dhabi?", "answer": "Al Reem Island is considered one of the best areas for waterfront living in Abu Dhabi, offering high-rise apartments with sea views and promenade access."},
        {"question": "What are the best residential towers in Downtown Dubai?", "answer": "Iconic residential towers in Downtown Dubai include Burj Khalifa Residences, The Address Residences, Forte, Opera Grand, and Boulevard Heights, offering luxury living and city views."},
        {"question": "Which towers in Dubai Marina have the highest rental demand?", "answer": "Towers like Marina Gate, Princess Tower, and JBR (Jumeirah Beach Residence) often have high rental demand due to their prime location, amenities, and proximity to transportation and entertainment."},
        {"question": "What are the tallest residential towers in Dubai?", "answer": "Some of the tallest residential towers in Dubai include Princess Tower, 23 Marina, Elite Residence, and Cayan Tower, offering panoramic views of the city and coastline."},
        {"question": "Which towers in Abu Dhabi are most popular for expats?", "answer": "Popular towers for expats in Abu Dhabi include those in Al Reem Island (e.g., Gate Towers, Sun & Sky Towers), and newly developed residences on Yas Island and Saadiyat Island."},
        {"question": "Are there any branded residences in towers?", "answer": "Yes, Dubai has numerous branded residences, often in collaboration with luxury fashion houses or hotel chains, such as Armani Residences, Versace Residences, Address Residences, and Bvlgari Residences."},
        {"question": "What zones allow expats to buy property in Sharjah?", "answer": "Expats can buy property in designated development zones in Sharjah, primarily in master-planned communities like Aljada, Masaar, and Maryam Island, under a 100-year leasehold."},
        {"question": "Is Aljada a good place to invest in Sharjah?", "answer": "Yes, Aljada is considered a strong investment location in Sharjah due to its master-planned community features, including schools, retail, and entertainment, attracting both residents and tenants."},
        {"question": "Al Reem Island vs. Saadiyat Island – which is better for lifestyle?", "answer": "Al Reem Island offers a more urban, bustling lifestyle with high-rise apartments and amenities, while Saadiyat Island provides a luxurious, cultural, and beach-focused lifestyle with villas and low-rise apartments near cultural institutions."},
        {"question": "Is Downtown Dubai freehold?", "answer": "Yes, Downtown Dubai is a prime freehold area in Dubai, open for property ownership by all nationalities."},
        {"question": "Can expats own property in Dubai Marina?", "answer": "Yes, Dubai Marina is one of the most popular freehold areas where expats can own properties with full ownership rights."},
        {"question": "Can non-UAE citizens buy property in Dubai Hills Estate?", "answer": "Yes, Dubai Hills Estate is a freehold community in Dubai, and non-UAE citizens are permitted to purchase properties there."},
        {"question": "Is Dubai Hills good for family living?", "answer": "Yes, Dubai Hills Estate is highly regarded as a family-friendly community, offering villas, townhouses, green spaces, schools, and recreational facilities."},
        {"question": "Is Business Bay freehold for foreigners?", "answer": "Yes, Business Bay is a designated freehold area in Dubai, allowing foreigners to purchase both residential and commercial properties."},
        {"question": "Is JVC a freehold area?", "answer": "Yes, Jumeirah Village Circle (JVC) is a designated freehold area in Dubai, popular among expats and investors for its affordable properties and community feel."},
        {"question": "Can expats buy in Dubai Creek Harbour?", "answer": "Yes, Dubai Creek Harbour is a freehold master development where expats can purchase apartments and other properties."},
        {"question": "Is Dubai South a freehold zone?", "answer": "Yes, Dubai South is a freehold development, offering residential properties where foreigners can own with 100% ownership."},
        {"question": "What is Bayview by Address in Emaar Beachfront?", "answer": "Bayview is a new luxury residential development by Emaar, part of the Address Hotels + Resorts brand, located within the exclusive Emaar Beachfront community."},
        {"question": "What are the unit types in Bayview?", "answer": "Bayview by Address offers a range of luxury apartments, typically from 1 to 4 bedrooms, and exclusive penthouses, all with premium finishes and sea views."},
        {"question": "What is the Morocco Cluster in Damac Lagoons?", "answer": "Themed around Moroccan architecture with lush courtyards and lagoon-facing villas."},
        {"question": "What is Saadiyat Lagoons?", "answer": "An eco-conscious, luxury villa community launched by Aldar on Saadiyat Island."},
        {"question": "Can expats buy in Saadiyat Lagoons?", "answer": "Yes. It's a designated freehold zone open to all nationalities."},
        {"question": "What is Gardenia Bay in Yas Island?", "answer": "Waterfront apartments with smart technology and resort-style living by Aldar."},
        {"question": "What is new in Reem Hills Phase 2?", "answer": "Larger plot sizes, private pools, and upgraded modern villa designs."},
        {"question": "What’s the newest launch in Aljada Sharjah?", "answer": "Boulevard 5 and Nest 2 — offering smart studios, 1BR and 2BR apartments with retail below."},
        {"question": "What’s new on Maryam Island in 2025?", "answer": "Sapphire Residences, with full sea-view apartments and access to beach promenade."},
        {"question": "What is Damac Lagoons?", "answer": "A themed villa community with artificial lagoons, clustered by countries like Morocco, Venice, and Ibiza."},
        {"question": "Is Sobha Hartland a freehold community?", "answer": "Yes. Located in MBR City with luxury waterfront apartments and villas."},
        {"question": "Can foreigners buy property in Bluewaters or Port de La Mer?", "answer": "Yes. Meraas developments are 100% freehold and open to all nationalities."},
        {"question": "Are DP communities good for families?", "answer": "Yes. Communities like Mudon and Villanova are gated and designed for family living."},
        {"question": "Can expats buy in Aldar communities?", "answer": "Yes. Aldar offers freehold property to all nationalities in designated zones."},
        {"question": "Can foreigners buy in Arada’s projects?", "answer": "Yes. Arada offers 100-year leasehold property ownership with full resale and rental rights."},
        {"question": "What is Maryam Island by Eagle Hills?", "answer": "A waterfront community in Sharjah offering beachfront apartments and retail promenade."},
    ],
    "Off-Plan Property Questions": [
        {"question": "What are off-plan properties and are they safe to buy in the UAE?", "answer": "Off-plan properties are units bought before or during construction. They are generally safe in the UAE due to strict DLD (Dubai Land Department) regulations and escrow accounts that protect buyer funds."},
        {"question": "Is buying off-plan property safe in the UAE?", "answer": "Buying off-plan is generally considered safe in the UAE, especially with reputable developers, due to regulations like mandatory escrow accounts and DLD oversight that protect buyer investments."},
        {"question": "What is the difference between off-plan and ready property in Dubai?", "answer": "Off-plan refers to properties bought directly from the developer before or during construction, often with flexible payment plans. Ready property is already built and ready for immediate occupancy or rental."},
        {"question": "Is it safe to buy off-plan property in the UAE?", "answer": "Yes, it is considered safe due to strong regulations by RERA and DLD, including escrow accounts for buyer funds and strict project registration requirements for developers."},
        {"question": "What are common off-plan payment plans in Dubai?", "answer": "Common off-plan payment plans include construction-linked plans (e.g., 60/40, 70/30, 50/50), where payments are tied to construction milestones, and sometimes post-handover plans."},
        {"question": "What is a post-handover payment plan?", "answer": "A post-handover payment plan allows buyers to pay a portion of the property's price after receiving the keys (handover), often over 1-5 years, providing financial flexibility."},
        {"question": "How long does it take to receive handover after project completion?", "answer": "Handover typically occurs shortly after project completion and final inspections, usually within a few weeks to a couple of months, depending on the developer and individual unit readiness."},
        {"question": "What is the Oqood certificate in Dubai real estate?", "answer": "An Oqood certificate is a temporary registration document issued by the DLD for off-plan properties. It acts as proof of purchase until the final title deed is issued upon handover."},
        {"question": "What is the escrow account law for off-plan properties?", "answer": "UAE law mandates that all funds from off-plan property sales must be deposited into a DLD-controlled escrow account. Developers can only draw funds from this account based on construction progress, protecting buyer investments."},
        {"question": "What are off-plan properties in Dubai?", "answer": "Off-plan properties in Dubai are real estate units sold by developers that are either currently under construction or planned for future development, purchased before completion."},
        {"question": "Why do investors buy off-plan properties in Dubai?", "answer": "Investors buy off-plan for lower entry prices, flexible payment plans, potential for capital appreciation by handover, and the opportunity to customize finishes or choose prime units."},
        {"question": "What are the risks of buying off-plan property in the UAE?", "answer": "Risks include potential delays in handover, changes in market conditions, and sometimes alterations to unit specifications, though DLD regulations mitigate many of these risks."},
        {"question": "How do I check if an off-plan project is registered with the Dubai Land Department (DLD)?", "answer": "You can verify a project's registration and developer status using the official Dubai REST app or by visiting the DLD website, which lists all approved projects and escrow accounts."},
        {"question": "What is the minimum down payment for off-plan projects in Dubai?", "answer": "Minimum down payments for off-plan projects typically range from 5% to 20% of the property value, with the remainder spread across construction milestones or post-handover."},
        {"question": "Which are the best off-plan communities in Dubai in 2025?", "answer": "Top off-plan communities in 2025 include The Oasis, Palm Jebel Ali, Dubai Creek Harbour, Emaar South, Damac Lagoons, and Sobha Hartland II, offering new launches and growth potential."},
        {"question": "Can I sell an off-plan property in Dubai before completion?", "answer": "Yes, you can sell an off-plan property before completion, typically after paying a certain percentage (e.g., 30-40%) to the developer. This is known as an assignment or 'resale' of the Oqood."},
        {"question": "How long does it take to hand over an off-plan property in Dubai?", "answer": "Handover timelines vary by project but generally range from 2.5 to 4 years from the launch date for new off-plan developments."},
        {"question": "Can off-plan property buyers apply for a mortgage in Dubai?", "answer": "Yes, many banks in Dubai offer financing for off-plan properties, usually requiring a higher down payment than ready properties and disbursing funds in stages according to construction progress."},
        {"question": "Are off-plan projects available for expat ownership in Abu Dhabi?", "answer": "Yes, Abu Dhabi offers numerous off-plan projects for expat ownership, particularly in designated investment zones like Yas Island, Saadiyat Island, and Al Reem Island."},
        {"question": "What are the best upcoming off-plan communities in Abu Dhabi?", "answer": "Key upcoming off-plan communities in Abu Dhabi include new phases in Saadiyat Lagoons, Gardenia Bay on Yas Island, and various developments on Al Reem and Jubail Islands."},
        {"question": "Is there escrow protection for off-plan property buyers in Abu Dhabi?", "answer": "Yes, similar to Dubai, Abu Dhabi has regulations requiring developers to use escrow accounts for off-plan projects to safeguard buyer funds."},
        {"question": "Can I resell off-plan property before completion in Abu Dhabi?", "answer": "Yes, reselling off-plan property before completion is generally possible in Abu Dhabi, subject to developer terms and a certain percentage of payment already made."},
        {"question": "Can foreigners buy off-plan property in Sharjah?", "answer": "Yes, foreigners can buy off-plan property in designated leasehold communities in Sharjah, such as Arada's Aljada and Masaar developments."},
        {"question": "Are there payment plans for Sharjah off-plan projects?", "answer": "Yes, developers in Sharjah typically offer attractive off-plan payment plans, often construction-linked, similar to those in Dubai, making properties accessible."},
        {"question": "Do Sharjah off-plan properties come with a completion guarantee?", "answer": "While regulations aim for timely completion, guarantees usually relate to buyer fund protection via escrow. Specific completion guarantees vary by developer and contract."},
        {"question": "What happens if a Dubai off-plan project gets delayed?", "answer": "If an off-plan project is delayed beyond the agreed-upon handover date, buyers typically have contractual rights, which can range from compensation to the option to cancel the contract and receive a refund, depending on the SPA terms and DLD regulations."},
        {"question": "Can I cancel my off-plan purchase?", "answer": "Yes, under certain conditions, such as significant delays or developer breaches of contract. However, cancelling without valid grounds can incur penalties, usually forfeiture of a percentage of payments made."},
        {"question": "What is the cost of transferring an off-plan unit to another buyer?", "answer": "Transferring an off-plan unit (assignment) involves DLD transfer fees (4% of property value), a developer's NOC fee, and potentially a broker commission. The total cost can be around 5-6% of the property value."},
        {"question": "How do I check the construction status of an off-plan property?", "answer": "You can check the construction status through the DLD's REST app, the developer's official website or app, or by requesting updates from your broker or the developer directly."},
        {"question": "What documents should I receive when buying off-plan in Dubai?", "answer": "You should receive a reservation agreement, a Sale and Purchase Agreement (SPA) / Form F, the Oqood certificate, and detailed payment schedules."},
        {"question": "What is the role of the escrow account in off-plan deals?", "answer": "The escrow account, managed by a DLD-approved bank, holds buyer funds for off-plan properties. Funds are released to the developer in stages, tied to construction progress, ensuring buyer protection."},
        {"question": "Who pays the DLD registration fee for off-plan property?", "answer": "The DLD registration fee (4% of property value) is typically paid by the buyer, though some developers offer waivers or absorb this cost as part of launch promotions."},
        {"question": "Can I rent an off-plan property immediately after handover?", "answer": "Yes, once you receive the keys and the title deed is issued, you can immediately rent out your property, either long-term or as a short-term holiday home (with appropriate licensing)."},
        {"question": "What is the title deed issuance process after off-plan handover?", "answer": "After full payment and handover, the developer will coordinate with the DLD to transfer the Oqood certificate into a final title deed in your name, which typically takes 1-2 weeks."},
        {"question": "Are there payment plans for The Oasis project?", "answer": "Yes, Emaar typically offers attractive construction-linked payment plans for The Oasis, often with post-handover options."},
        {"question": "What is the expected completion date for Palm Jebel Ali villas?", "answer": "Palm Jebel Ali is a long-term project, with initial villa handovers expected to be several years out, likely from 2027-2030 onwards, depending on the phase."},
        {"question": "Are post-handover payment plans available?", "answer": "Yes. Damac offers 1–3 year post-handover plans on selected units."},
        {"question": "What is the starting price of apartments in Gardenia Bay?", "answer": "Studios start from around AED 765,000 with flexible payment plans."},
        {"question": "Are new off-plan projects in Dubai safe to invest in?", "answer": "Yes, if launched by top developers like Emaar, Nakheel, Aldar, and Arada, and registered with escrow accounts."},
        {"question": "Do new off-plan projects come with DLD waivers?", "answer": "Many developers offer limited-time 4% DLD fee waivers as launch promotions."},
        {"question": "How long are new off-plan project handover timelines?", "answer": "Most new projects launched in 2024–2025 aim for 2.5 to 4-year delivery."},
        {"question": "Which new off-plan communities offer post-handover payment plans?", "answer": "Damac Lagoons, Emaar South, Sobha Hartland II, and Gardenia Bay all offer post-handover options."},
        {"question": "Do UAE developers offer post-handover payment plans?", "answer": "Yes, especially for off-plan launches. Many offer 60/40 or 70/30 with up to 3-year post-handover terms."},
        {"question": "Does Emaar offer payment plans?", "answer": "Yes. Typical off-plan plans are 80/20 or 90/10 with flexible post-handover options."},
        {"question": "Do Sobha projects offer post-handover plans?", "answer": "Yes. Many projects offer 60/40 plans and high-end finishes."},
        {"question": "Are Ellington projects off-plan or ready?", "answer": "Primarily off-plan, but with fast delivery and award-winning architecture."},
        {"question": "Do all off-plan projects require an escrow account?", "answer": "Yes. All off-plan developments must have a DLD-monitored escrow account to protect buyers."},
        {"question": "What if a developer delays the off-plan project delivery?", "answer": "Buyers are protected via DLD escrow laws. In severe cases, RERA can cancel projects and refund from escrow."},
        {"question": "Can I cancel my off-plan purchase?", "answer": "Yes, under certain conditions (delays, breaches). However, a penalty may apply depending on the SPA."},
        {"question": "Can I sell an off-plan property before handover?", "answer": "Yes. This is called an assignment sale — subject to developer approval and minimum payment thresholds (often 30–40%)."},
        {"question": "Can I get a mortgage for an off-plan property in Dubai?", "answer": "Yes. Some banks offer off-plan project financing — especially for Emaar, Nakheel, Aldar, etc., with 50–70% LTV."},
        {"question": "Can I get a refund if the developer fails to deliver the project?", "answer": "Yes. RERA may refund buyers from escrow accounts if a project is officially canceled."},
        {"question": "Are buyers protected when buying in newly launched projects?", "answer": "Yes. Projects must be DLD-approved, escrow-regulated, and supervised under Law No. 8 of 2007."},
        {"question": "Are off-plan properties cheaper than ready units?", "answer": "Yes. Off-plan units are often 10–20% cheaper and come with flexible payment plans."},
        {"question": "What is the starting price of new off-plan projects in Dubai?", "answer": "Studio: AED 500K – AED 800K; 1BR: AED 900K – AED 1.4M; 2BR: AED 1.5M – AED 2.2M; 3BR Villas: AED 1.9M – AED 3.5M."},
        {"question": "What is the price per sq.ft. for off-plan in Creek Harbour and Beachfront?", "answer": "Emaar Beachfront: AED 3,000–4,200/sq.ft.; Creek Harbour: AED 1,600–2,200/sq.ft."},
        {"question": "Are there off-plan properties under AED 1 million in 2025?", "answer": "Yes. Projects in JVC, Dubai South, Arjan, Dubailand, and Town Square offer studios and 1BRs under AED 1M."},
    ],
    "Developer-Specific FAQs": [
        {"question": "Who are the top off-plan developers in Abu Dhabi?", "answer": "Aldar Properties is the leading off-plan developer in Abu Dhabi, known for large-scale projects on Yas and Saadiyat Islands, alongside other developers like Eagle Hills and Imkan."},
        {"question": "Who are the main developers offering off-plan in Sharjah?", "answer": "Arada is the largest private developer in Sharjah (Aljada, Masaar), along with Eagle Hills (Maryam Island) and Omran Properties."},
        {"question": "What is The Oasis by Emaar?", "answer": "The Oasis is a new ultra-luxury villa and mansion community by Emaar Properties, designed around canals and green spaces, offering high-end living."},
        {"question": "Where is The Oasis located in Dubai?", "answer": "The Oasis is strategically located near Sheikh Zayed Bin Hamdan Al Nahyan Street, close to Emirates Road and major attractions like Al Maktoum Airport."},
        {"question": "What property types are available in The Oasis?", "answer": "The Oasis primarily offers luxury villas and grand mansions with varying configurations, catering to high-net-worth individuals."},
        {"question": "What is Sobha Hartland II?", "answer": "Sobha Hartland II is a new master-planned community by Sobha Realty, offering luxury villas, townhouses, and apartments with crystal lagoons and extensive green spaces."},
        {"question": "Are the lagoons real and swimmable?", "answer": "Yes, Sobha Hartland II features real, swimmable crystal lagoons and beaches within the community for residents."},
        {"question": "What is Fairway Villas 3 in Emaar South?", "answer": "The latest golf-course-facing villas within Emaar South, near Dubai World Central."},
        {"question": "Is Emaar South a good location to invest now?", "answer": "Yes. Proximity to Al Maktoum Airport and Expo City boosts future ROI."},
        {"question": "Who are the most trusted developers in Dubai?", "answer": "The most reputable developers include Emaar, Nakheel, Meraas, Dubai Properties, Sobha, and Damac."},
        {"question": "What guarantees do UAE developers offer for off-plan projects?", "answer": "Most offer 1-year snagging warranties and 10-year structural warranties, in line with RERA regulations."},
        {"question": "Who is Emaar Properties?", "answer": "Emaar is Dubai’s largest master developer, known for Downtown Dubai, Burj Khalifa, Dubai Hills, and Emaar Beachfront."},
        {"question": "What makes Emaar popular among investors?", "answer": "Emaar is known for timely handovers, strong resale value, high-quality finishing, and access to prime locations."},
        {"question": "What are some new projects by Emaar in 2025?", "answer": "The Oasis, Bayview, Palmiera Villas in The Oasis, and new towers in Creek Harbour and Dubai Hills."},
        {"question": "Is Damac a reliable real estate developer?", "answer": "Yes. Damac is known for large-scale themed communities like Damac Hills, Damac Lagoons, and branded towers."},
        {"question": "Does Damac offer branded residences?", "answer": "Yes. Damac has partnered with Cavalli, de GRISOGONO, and Versace for branded towers."},
        {"question": "What is Nakheel famous for?", "answer": "Nakheel is the developer behind iconic Palm Jumeirah, Palm Jebel Ali, and The World Islands."},
        {"question": "Is Palm Jebel Ali developed by Nakheel?", "answer": "Yes. It’s their relaunch of the massive artificial island with beachfront villas and signature fronds."},
        {"question": "What are Nakheel’s current projects?", "answer": "Palm Jebel Ali, Jebel Ali Village Villas, and developments in Dubai Islands."},
        {"question": "What makes Sobha Realty unique in Dubai?", "answer": "Sobha offers in-house construction and premium build quality, especially in Sobha Hartland and Hartland II."},
        {"question": "What are Meraas’ major Dubai developments?", "answer": "City Walk, Bluewaters Island, Jumeirah Bay Island, and Port de La Mer."},
        {"question": "Are Meraas projects considered luxury?", "answer": "Yes. Their portfolio is focused on premium urban beachfront and designer developments."},
        {"question": "Who owns Dubai Properties?", "answer": "It’s part of Dubai Holding, the government-owned master developer."},
        {"question": "What are some key projects by DP?", "answer": "Jumeirah Beach Residence (JBR), Business Bay, Mudon, Villanova, and Dubai Wharf."},
        {"question": "Is Ellington a good developer for boutique buyers?", "answer": "Yes. They focus on design-driven apartments in JVC, Downtown, and Palm Jumeirah."},
        {"question": "Is Aldar a government-backed developer?", "answer": "Yes. It’s the largest master developer in Abu Dhabi, known for Yas Island, Saadiyat Island, and Al Ghadeer."},
        {"question": "What new projects has Aldar launched in 2025?", "answer": "Gardenia Bay, Saadiyat Lagoons Phase 2, and new launches on Reem and Jubail Islands."},
        {"question": "Who is Arada?", "answer": "Arada is Sharjah’s largest private developer, behind Aljada, Masaar, and Nest student housing."},
        {"question": "Is Aljada good for investment in 2025?", "answer": "Yes. It’s a master community with schools, hotels, malls, and residential blocks ideal for rental yield."},
        {"question": "Is Eagle Hills a reputable developer?", "answer": "Yes. It’s based in Abu Dhabi and has delivered successful mixed-use projects across the UAE and MENA region."},
        {"question": "Which Dubai developer offers the best post-handover plans?", "answer": "Damac and Emaar often provide flexible 2–4 year post-handover payment plans."},
        {"question": "Which developer has the best ROI properties in 2025?", "answer": "Damac (affordable high-yield units), Emaar (long-term capital appreciation), and Sobha (high-end end-user demand)."},
        {"question": "Who builds the best quality luxury villas in Dubai?", "answer": "Sobha Realty and Emaar are top contenders in terms of construction and finishing quality."},
        {"question": "Are branded residences only built by certain developers?", "answer": "Yes. Damac (Cavalli, Versace), Emaar (Address, Vida), and Omniyat (Dorchester, One Palm) lead branded segments."},
        {"question": "Which developer is ideal for first-time investors?", "answer": "Dubai Properties, Ellington, and Arada offer lower price points with high rental demand."},
    ],
    "Legal FAQs": [
        {"question": "How do I verify if a developer is registered in Dubai?", "answer": "You can verify developer registration and project approval status through the official Dubai Land Department (DLD) website or their Dubai REST app."},
        {"question": "Is it mandatory to register a property purchase with the DLD?", "answer": "Yes, all property purchases in Dubai must be registered with the Dubai Land Department (DLD) to ensure legal ownership and transparency."},
        {"question": "How do I transfer property ownership in the UAE?", "answer": "Property ownership transfer involves completing documentation, obtaining NOC from the developer (if applicable), paying DLD transfer fees, and finally registering the new owner with the DLD."},
        {"question": "How can I verify if a UAE developer is approved?", "answer": "Use the Dubai REST App or visit the official DLD or RERA portal to confirm developer and project status."},
        {"question": "What is RERA in Dubai?", "answer": "The Real Estate Regulatory Agency (RERA) regulates developers, brokers, and property transactions under DLD."},
        {"question": "How do I verify a project's legal status in Dubai?", "answer": "Use the Dubai REST app to check project approvals, escrow accounts, and developer licensing."},
        {"question": "Is it mandatory to use a registered real estate broker?", "answer": "Yes. Brokers must hold valid RERA cards and be licensed under the Dubai Economic Department."},
        {"question": "Are real estate disputes handled in court or arbitration?", "answer": "Disputes are typically handled via Dubai Rental Disputes Centre (RDC) or Property Courts, depending on the case type."},
        {"question": "Are all developers required to register their project?", "answer": "Yes. Developers must register with RERA and DLD and link each project to an escrow account."},
    ],
    "Pricing & ROI": [
        {"question": "Which is better: apartment or villa for investment in Dubai?", "answer": "Apartments generally offer higher rental yields and lower entry prices, while villas tend to provide higher capital appreciation over the long term and better family living."},
        {"question": "Which Dubai communities offer the best rental returns?", "answer": "Areas like Jumeirah Village Circle (JVC), Dubai Silicon Oasis, International City, and Business Bay often offer strong rental returns for apartments. Villas in Arabian Ranches and Mudon also perform well."},
        {"question": "Are service charges high in Sharjah properties?", "answer": "Service charges in Sharjah properties are generally lower compared to Dubai, reflecting the overall lower cost of living and property prices in the emirate."},
        {"question": "What is the average ROI for property investment in Dubai?", "answer": "The average ROI (Return on Investment) for residential properties in Dubai typically ranges from 5% to 8% annually, depending on the property type, location, and market conditions."},
        {"question": "Which areas in Dubai have the highest rental yields?", "answer": "Areas known for high rental yields include Jumeirah Village Circle (JVC), International City, Dubai Silicon Oasis, and Business Bay (for apartments); and Mudon, and Arabian Ranches (for villas)."},
        {"question": "Is buying a villa or apartment better for long-term capital gain in the UAE?", "answer": "Villas, particularly in established family communities and prime locations, tend to show stronger long-term capital appreciation due to limited supply and high demand from end-users."},
        {"question": "How do I calculate ROI on a UAE property investment?", "answer": "ROI is calculated as (Annual Rental Income - Annual Expenses) / Total Property Purchase Price. For capital appreciation, it's (Sale Price - Purchase Price) / Purchase Price."},
        {"question": "Which emirate offers the best value for real estate investors?", "answer": "Sharjah often offers the best value in terms of price per square foot and potentially higher rental yields for affordable properties, while Dubai offers higher capital appreciation in prime areas."},
        {"question": "Which is better: Downtown Dubai or Business Bay for investment?", "answer": "Downtown Dubai offers premium luxury and higher capital appreciation potential, while Business Bay provides slightly more affordable options with strong rental yields, especially for business professionals."},
        {"question": "JVC vs. Arjan – which area is better for budget investors?", "answer": "Both offer good value. JVC is more established with a community feel, while Arjan is newer, often offering highly competitive prices and good rental returns for studios and 1-beds."},
        {"question": "What are annual service charges in Dubai?", "answer": "Annual service charges (or maintenance fees) in Dubai are recurring fees paid by property owners for the upkeep and management of common areas, amenities, and infrastructure within a building or community."},
        {"question": "Who pays the service charges: owner or tenant?", "answer": "Service charges are typically paid by the property owner, not the tenant. Tenants pay rent, while owners cover maintenance and community fees."},
        {"question": "What is a chiller-free apartment in Dubai?", "answer": "A 'chiller-free' apartment means that the tenant or owner does not pay separately for district cooling (AC usage), as the cost is included in the service charges or master developer fees, reducing utility bills."},
        {"question": "How do I estimate maintenance costs for villas in Dubai?", "answer": "Villa maintenance costs include service charges (for common areas), potentially district cooling fees, garden/pool maintenance (if private), and general repairs. Budget 1-2% of property value annually."},
        {"question": "Are service charges lower in Sharjah or Abu Dhabi compared to Dubai?", "answer": "Generally, service charges in both Sharjah and Abu Dhabi tend to be lower than in Dubai, contributing to a potentially lower overall cost of ownership."},
        {"question": "Is now a good time to buy property in Dubai?", "answer": "As of 2025, Dubai's property market is experiencing stable growth due to high demand, population influx, and government initiatives, making it a favorable time for strategic investment, especially in prime and emerging areas."},
        {"question": "What is driving property prices in Dubai?", "answer": "Key drivers include population growth, strong economic performance, strategic government initiatives (like Golden Visas), high demand from international investors, limited supply of luxury villas, and Dubai's safe-haven status."},
        {"question": "Will Dubai property prices drop in 2025?", "answer": "Most experts predict continued moderate growth or stability for Dubai property prices in 2025, rather than a significant drop, driven by ongoing demand and new developments."},
        {"question": "Which areas in the UAE are expected to grow the most by 2026?", "answer": "Areas around Expo City Dubai, new master communities like The Oasis and Palm Jebel Ali, and established but still developing areas like Dubai Hills Estate and Dubai Creek Harbour are expected to see significant growth."},
        {"question": "Are property prices rising in Sharjah and Abu Dhabi too?", "answer": "Yes, property prices in Abu Dhabi and Sharjah are also experiencing steady growth, albeit at a more moderate pace than Dubai, driven by new project launches and increasing population."},
        {"question": "What is the average price for off-plan apartments in Sharjah?", "answer": "Average prices for off-plan apartments in Sharjah typically start from AED 450,000 for studios and range up to AED 1.2 million for 2-bedroom units, offering good value compared to Dubai."},
        {"question": "Are off-plan investments more profitable than ready properties?", "answer": "Off-plan investments can offer higher potential capital appreciation if the market grows during construction, and they come with more flexible payment plans, often resulting in a higher ROI by handover if timed well."},
        {"question": "What is the average price per sq.ft in Downtown Dubai (2025)?", "answer": "As of 2025, the average price per sq.ft. in Downtown Dubai ranges from AED 2,500 to AED 4,000, varying significantly by building, view, and finishing quality."},
        {"question": "How does Business Bay compare to Downtown Dubai in ROI?", "answer": "Business Bay generally offers slightly higher rental yields (ROI) due to more competitive prices and strong demand from working professionals, while Downtown Dubai often sees higher capital appreciation on higher-value properties."},
        {"question": "Is JVC a good investment location in 2025?", "answer": "Yes, JVC continues to be a good investment location in 2025, offering strong rental yields for apartments and townhouses due to its affordability and central location, attracting both tenants and first-time buyers."},
        {"question": "What is the average price of villas in Palm Jebel Ali?", "answer": "Villas in Palm Jebel Ali are launching at premium prices, likely starting from AED 15-20 million for standard units, with larger and more exclusive mansions significantly higher."},
        {"question": "Is Bayview a good investment in 2025?", "answer": "Yes, Bayview in Emaar Beachfront is considered a strong investment due to its prime waterfront location, luxury branding (Address Hotels), and potential for high capital appreciation and rental yields."},
        {"question": "Is Maryam Island a good location for short-term rental?", "answer": "Yes. Its central waterfront location and urban design attract tourists and professionals alike."},
        {"question": "Are Damac projects good for rental ROI?", "answer": "Yes. Areas like Damac Hills and Lagoons offer 6–8% ROI due to affordability and demand."},
        {"question": "What is the average price of property in Dubai in 2025?", "answer": "As of 2025, the average price is around AED 1,450–1,750 per sq.ft., depending on location, developer, and type."},
        {"question": "What is the current price trend in Dubai real estate?", "answer": "Prices are increasing moderately in prime areas and master communities, especially for villas and branded residences."},
        {"question": "Are property prices going up or down in Dubai?", "answer": "Prices are stable to rising in most communities due to population growth, investor demand, and Golden Visa incentives."},
        {"question": "What are the most expensive areas to buy property in Dubai?", "answer": "Palm Jumeirah: AED 3,500–7,000/sq.ft.; Downtown Dubai: AED 2,500–4,000/sq.ft.; Dubai Hills Estate – Golf Front: AED 2,000–3,000/sq.ft.; Jumeirah Bay Island: AED 6,000–8,000/sq.ft."},
        {"question": "What are the most affordable areas to buy property in Dubai?", "answer": "Dubai South (Emaar South, MAG): AED 750–1,000/sq.ft.; Jumeirah Village Circle (JVC): AED 950–1,250/sq.ft.; International City: AED 500–750/sq.ft."},
        {"question": "What is the price of an apartment in Dubai Marina?", "answer": "Prices range from AED 1,600 to 3,000/sq.ft., depending on tower, view, and floor."},
        {"question": "What is the average villa price in Dubai?", "answer": "3BR Townhouse: AED 1.8M – AED 2.6M; 4BR Villa: AED 2.8M – AED 5.5M; Luxury Signature Villas: AED 15M – AED 250M+"},
        {"question": "What is the cheapest community to buy a villa in Dubai?", "answer": "Dubailand (Villanova, Rukan): From AED 1.6M; DAMAC Lagoons: 3BR from AED 1.85M; Emaar South: Starting AED 1.75M."},
        {"question": "What are the prices of 10-bedroom mansions in Emirates Hills?", "answer": "From AED 80M up to AED 250M+, depending on plot size, golf view, and customization."},
        {"question": "What is the average price of property in Abu Dhabi in 2025?", "answer": "Apartments: AED 1,100–1,500/sq.ft.; Villas: AED 1,200–2,500/sq.ft.; Premium Islands (Saadiyat, Yas): AED 2,800–4,500/sq.ft."},
        {"question": "What is the price of Aldar’s Saadiyat Lagoons Villas?", "answer": "4–5BR villas range from AED 6M to AED 12M+, depending on plot and lagoon view."},
        {"question": "Is property cheaper in Sharjah than in Dubai?", "answer": "Yes. Sharjah offers 60–70% lower prices, starting from AED 500–900/sq.ft. in leasehold communities."},
        {"question": "What are Arada’s project prices in Aljada or Masaar?", "answer": "Apartments in Aljada: From AED 450K; Villas in Masaar: From AED 1.7M; Townhouses: From AED 1.4M."},
        {"question": "Can investors get ROI in Sharjah like Dubai?", "answer": "Yes. Yields in new Sharjah communities range from 6–9%, especially in Arada or Eagle Hills developments."},
        {"question": "Are resale prices higher than off-plan prices in Dubai?", "answer": "In popular areas, off-plan properties see 15–35% value growth by handover."},
        {"question": "Which areas in Dubai are giving the best capital appreciation?", "answer": "Dubai Creek Harbour, Dubai Hills Estate, Palm Jebel Ali, Damac Lagoons (due to early-phase pricing)."},
        {"question": "What is the ROI in Dubai rental market in 2025?", "answer": "Apartments: 6% – 8.5%; Villas: 5% – 7%; Short-term rentals: 8% – 12% in tourist zones."},
        {"question": "Are Dubai property prices expected to rise further in 2025–26?", "answer": "Yes. Continued investor migration, limited villa supply, and mega launches like Palm Jebel Ali will drive demand and prices."},
    ],
    "Visa, Tax & Inheritance": [
        {"question": "What is the minimum property investment for a Golden Visa in Abu Dhabi?", "answer": "In Abu Dhabi, a minimum property investment of AED 2 million is required for eligibility for the 10-year Golden Visa."},
        {"question": "What is the Golden Visa real estate requirement in the UAE?", "answer": "To qualify for a UAE Golden Visa through real estate, you typically need to invest a minimum of AED 2 million (approx. $545,000 USD) in property."},
        {"question": "Are there property taxes in Dubai, Abu Dhabi, or Sharjah?", "answer": "No, the UAE generally does not impose annual property taxes, capital gains tax on real estate, or inheritance tax, making it an attractive investment destination."},
        {"question": "Can you get residency by buying property in the UAE?", "answer": "Yes, purchasing property in the UAE can make you eligible for a residency visa, with the Golden Visa being the most prominent option for investments of AED 2 million or more."},
        {"question": "Can property ownership in the UAE lead to long-term residency?", "answer": "Yes, property ownership can lead to long-term residency, particularly through the Golden Visa program for investments above AED 2 million."},
        {"question": "Is there property tax in Dubai or Abu Dhabi?", "answer": "No, neither Dubai nor Abu Dhabi imposes annual property taxes. The main fee is the DLD transfer fee upon purchase."},
        {"question": "Are off-plan properties in Dubai eligible for a Golden Visa?", "answer": "Yes, off-plan properties in Dubai can be eligible for the Golden Visa, provided the property value (or equity paid) meets the AED 2 million threshold, and the property is from an approved developer."},
        {"question": "What is the average rental yield for apartments in Dubai Marina?", "answer": "The average rental yield for apartments in Dubai Marina typically ranges from 5.5% to 7.5%, making it an attractive area for buy-to-let investors."},
        {"question": "What is the average rental yield for properties in Palm Jumeirah?", "answer": "Rental yields in Palm Jumeirah vary, but luxury apartments can see 4-6%, while high-end villas may offer 3-5% due to higher property values."},
        {"question": "What are the rental yields for properties in Dubai Creek Harbour?", "answer": "Dubai Creek Harbour, being a newer development, currently offers rental yields around 5-7% for apartments, with potential for growth as the community matures."},
        {"question": "What is the minimum property value for a 10-year Golden Visa?", "answer": "AED 2 million (approx. US$ 545,000)."},
        {"question": "Can I get a Golden Visa with a mortgage property?", "answer": "Yes. The mortgage value can count towards the AED 2M requirement, provided the bank provides an NOC."},
        {"question": "Is property ownership still a route for residency in Dubai?", "answer": "Yes. Investment in property is a direct pathway for residency, including the 10-year Golden Visa."},
        {"question": "What is the difference between an Investor Visa and a Golden Visa?", "answer": "An Investor Visa is typically 2-3 years, while the Golden Visa is 5 or 10 years for larger investments, offering more benefits and stability."},
        {"question": "Can I apply for a Golden Visa with off-plan property?", "answer": "Yes. If the off-plan property's value (or the amount paid to date) is AED 2 million or more."},
        {"question": "Does the UAE have property taxes?", "answer": "No. There are no annual property taxes, only DLD transfer fees (4%) at the time of purchase."},
        {"question": "Is there an inheritance tax on property in Dubai?", "answer": "No. The UAE does not impose inheritance tax on real estate."},
        {"question": "Do I need a local bank account to buy property in Dubai?", "answer": "Yes. It's essential for transactions. You can open a non-resident account or a local resident account."},
        {"question": "Can I buy property in Dubai without being a resident?", "answer": "Yes. Foreigners can purchase freehold property in designated areas without holding a residency visa."},
        {"question": "What are the visa options for property owners?", "answer": "You can get a 2-year Investor Visa for AED 750K+ property, or a 10-year Golden Visa for AED 2M+ property."},
        {"question": "What is the minimum property value to get a 10-year Golden Visa?", "answer": "AED 2 million. This can be a single property or multiple properties adding up to AED 2M."},
        {"question": "Can I apply for Golden Visa with a financed (mortgaged) property?", "answer": "Yes. You must provide a statement from the bank confirming the amount paid against the mortgage and an NOC from the bank."},
    ],
    "Rental & Property Management": [
        {"question": "What is the typical rental yield for apartments in Downtown Dubai?", "answer": "Rental yields for apartments in Downtown Dubai typically range from 4% to 6%, depending on the building, unit size, and amenities."},
        {"question": "What is the rental market like in Yas Island, Abu Dhabi?", "answer": "Yas Island has a strong rental market driven by its attractions and family-friendly amenities, with high demand for both apartments and villas, particularly from professionals working in the area."},
        {"question": "How do I rent out my property in Dubai?", "answer": "To rent out property in Dubai, you need to register the Ejari contract with DLD, ensure all documents are in order, and market the property through licensed agents. You may also consider property management services."},
        {"question": "What is Ejari in Dubai?", "answer": "Ejari is an online registration system by the DLD that legalizes tenancy contracts in Dubai. It's mandatory for all rental agreements and ensures transparency and legality between landlords and tenants."},
        {"question": "Is Ejari mandatory for all rental contracts in Dubai?", "answer": "Yes, Ejari registration is mandatory for all residential and commercial tenancy contracts in Dubai to ensure their legal validity and to be able to file a case at the Rental Disputes Center if needed."},
        {"question": "What are typical rental contract terms in Dubai?", "answer": "Typical rental contracts in Dubai are for one year, renewable, with rent paid in 1 to 4 cheques. Longer terms or different payment frequencies can be negotiated."},
        {"question": "Can a landlord increase rent in Dubai?", "answer": "Rent increases in Dubai are governed by RERA's rental index. Landlords can increase rent based on market value and after giving 90 days' notice, provided the increase is within the RERA calculator's limits."},
        {"question": "How much can a landlord increase rent in Dubai?", "answer": "The maximum permissible rent increase in Dubai is determined by the RERA Rent Calculator, which considers the current market rent. The increase depends on how far below market value the current rent is."},
        {"question": "What happens if a tenant wants to break a lease early in Dubai?", "answer": "Breaking a lease early typically involves penalties, as stipulated in the tenancy contract, usually forfeiture of one or two months' rent as compensation to the landlord."},
        {"question": "What are common landlord responsibilities in Dubai?", "answer": "Landlord responsibilities in Dubai include ensuring the property is habitable, carrying out major maintenance, providing quiet enjoyment of the property, and registering the Ejari contract."},
        {"question": "What is Ejari registration in Dubai?", "answer": "Ejari is the mandatory online registration system for all rental contracts in Dubai to legalize the tenancy relationship between landlords and tenants."},
    ],
    "Property Buying Process": [
        {"question": "What are the typical closing costs when buying property in Dubai?", "answer": "Closing costs in Dubai typically include the DLD (Dubai Land Department) transfer fee (4% of property value), DLD administration fees, property registration fees, and real estate agent commission (2% + VAT)."},
        {"question": "What are the DLD transfer fees in Dubai?", "answer": "The DLD (Dubai Land Department) transfer fee is 4% of the property value, paid by the buyer, though sometimes developers may offer to cover it for off-plan properties."},
        {"question": "What are the costs involved in buying a ready property in Dubai?", "answer": "DLD transfer fee (4% of property value), DLD admin fees (approx. AED 4,000-5,000), real estate agent commission (2% + 5% VAT), and potentially mortgage registration fees (0.25% of mortgage amount)."},
        {"question": "What are off-plan properties and are they safe to buy in the UAE?", "answer": "Off-plan properties are units bought before or during construction. They are generally safe in the UAE due to strict DLD (Dubai Land Department) regulations and escrow accounts that protect buyer funds."},
    ]
}

def ingest_categorized_faq_data():
    try:
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print("Connected to Qdrant.")

        print(f"Loading SentenceTransformer model: {EMBED_MODEL_ID}...")
        model = SentenceTransformer(EMBED_MODEL_ID)
        print("Model loaded.")

        vector_size = model.get_sentence_embedding_dimension()
        print(f"Embedding vector size: {vector_size}")

        if not client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' does not exist. Creating...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Collection '{COLLECTION_NAME}' created with vector size {vector_size} and Cosine distance.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists. Deleting existing points for fresh ingestion.")
            client.delete_collection(collection_name=COLLECTION_NAME) 
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            ) 
            print(f"Collection '{COLLECTION_NAME}' deleted and recreated for fresh ingestion.")

        texts_to_encode = []
        points_to_upsert_payloads = [] 

        for category, faqs in CATEGORIZED_FAQ_DATA.items():
            for faq in faqs:
                if not faq.get("question") or not faq.get("answer"):
                    print(f"Skipping malformed FAQ in category '{category}': {faq}")
                    continue

                full_text = f"Category: {category}. Question: {faq['question']}. Answer: {faq['answer']}"
                texts_to_encode.append(full_text)

                payload = {
                    "question": faq["question"],
                    "answer": faq["answer"],
                    "category": category 
                }
                points_to_upsert_payloads.append(payload)

        if not texts_to_encode:
            print("No valid FAQ data to encode. Exiting.")
            return

        print(f"Encoding {len(texts_to_encode)} FAQ texts into embeddings...")
        embeddings = model.encode(texts_to_encode, show_progress_bar=True).tolist()
        print("Encoding complete.")

        points_to_upsert = []
        for i, payload in enumerate(points_to_upsert_payloads):
            points_to_upsert.append(
                PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=embeddings[i], 
                    payload=payload 
                )
            )

        # 4.7. 
        print(f"Upserting {len(points_to_upsert)} points to collection '{COLLECTION_NAME}'...")
        operation_info = client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert,
            wait=True
        )
        print(f"Successfully ingested {len(points_to_upsert)} FAQs into Qdrant collection '{COLLECTION_NAME}'.")
        print(f"Qdrant Operation Info: {operation_info}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    ingest_categorized_faq_data()
    print("\nCategorized data ingestion complete.")
