import os
import json
import re
from tqdm import tqdm
import random
import requests
import torch
from tqdm.contrib.concurrent import thread_map
import concurrent.futures
import time
from openai import OpenAI
import argparse
import pdb

# Function to configure OpenAI client
def get_openai_client():
    client = OpenAI(
        api_key="Your API Key Here", 
        base_url="Your Base URL Here",
    )
    return client

def get_prompt(dataset_name, reviews):
    # System prompts for different dataset types
    system_prompts = {
        'CDs': """You are a helpful assistant specialized in analyzing causal relationships in music product reviews. 
Your task is to extract hidden factors or attributes that can be inferred from user reviews about CDs and music products.
Hidden factors are attributes not explicitly mentioned but can be inferred from context, like user preferences or item characteristics.
You must follow the exact output format specified and only output the factors without any additional text.""",

        'yelp': """You are a helpful assistant specializing in analyzing local business reviews. Your task is to extract key attributes about the business and inferred preferences of the user based on their review text and business data. The goal is to identify standardized factors that explain the user's experience.""",

        'office': """You are a helpful assistant specialized in analyzing causal relationships in office product reviews.
Your task is to extract hidden factors or attributes that explain why users purchased these office products.
Hidden factors are reasons for purchase not explicitly mentioned but can be inferred from context.
You must follow the exact output format specified and only output the factors without any additional text."""
    }

    # User prompts for different dataset types
    user_prompt_templates = {
        'CDs': """The following are the reviews of a user about various items. Please extract all possible causal hidden factors or attributes related to these items.

Below are some examples of hidden factors that you can consider:
1. Music genre/style (e.g., rock, pop, classical, jazz, electronic)
2. Artist/band popularity (e.g., famous singers, well-known bands)
3. Limited edition/collector's items (e.g., special edition, signed copies, deluxe version)
4. Price sensitivity (e.g., discounts, promotions, bundle offers)
5. Music Charts (e.g., Billboard Hot 100, UK Top 40) 
6. Sound quality (e.g., high-quality audio, 24-bit, stereo)
7. Music format (e.g., CD, vinyl, digital download code)
8. Brand reputation (e.g., trusted e-commerce platforms, specialty stores)
9. Social proof (e.g., user reviews, ratings, social media recommendations)
10. Thematic/Concept albums (e.g., movie soundtracks, culturally themed albums)
11. Fan culture (e.g., fan merchandise, artist-specific albums)
12. Release timing (e.g., new album releases, anniversary editions)
13. Collector's Mentality (e.g., limited edition, rare items)
14. Music Festival (e.g., Coachella, Glastonbury, Lollapalooza)
15. Music Awards (e.g., Grammy Awards, MTV Music Awards)
16. Gift buying (e.g., for birthdays, holidays, special occasions)

For example, the user may have good impression on Taylor Swift, so there are two edges like <user>[TOW]<Taylor Swift> and <item>[TOW]<country music>, so we could recommend him with other good country music.

You need to output all the Hidden Factors from above 16 examples or any other factors that you can infer from the reviews.

Here are the interactions (Note that the user's rating range from 1 to 5, 5 means the user likes the item very much, 1 means the user dislikes the item):
{reviews}

Please output in a structured format like this (You should replace the user_id and item_id with the actual user_id and items parent_asin):
1. <user>[TOW]<high-quality>
2. <item>[TOW]<Religious>
3. <user>[TOW]<Christianity>
4. <item>[TOW]<Queen>
5. <user>[TOW]<Wrap>
6. <item>[TOW]<Popularity>

AND DO NOT REPLY ANY OTHER INFORMATION.""",

'yelp':"""The following are the reviews of a user about various businesses. Please extract all relevant attributes and user preferences. You should try to normalize concepts from the review into the standard categories provided below.
        Here are some example categories to consider:
        1. Cuisine/Service Style (e.g., Italian, Mexican, Thai, vegan-friendly, brunch)
        2. Price Range/Value (e.g., budget-friendly, upscale, good value, expensive)
        3. Ambiance/Atmosphere (e.g., romantic, family-friendly, noisy, cozy, modern)
        4. Occasion Suitability (e.g., date night, group hangout, birthday celebration, quick bite)
        5. Service Quality (e.g., attentive staff, fast service, rude service)
        6. Location/Convenience (e.g., downtown, easy parking, near public transit)
        7. Dietary Accommodations (e.g., gluten-free options, vegetarian, allergy-friendly)
        8. Brand Reputation (e.g., famous chain, local favorite, Michelin-starred)
        9. Specific Features (e.g., outdoor seating, dog-friendly, live music)
        10. Food/Product Quality (e.g., fresh ingredients, authentic taste, artisanal)
        For example, if a user praises a quiet Italian restaurant for a birthday dinner, you could infer factors like <user_id>[TOW]<Celebratory Occasions> and <business_id>[TOW]<Romantic Ambiance>.
        Here are the interactions (Note that the user's rating range from 1 to 5, 5 means the user likes the business very much, 1 means the user dislikes the business):
        {reviews}
        Please output in a structured format like this (You MUST replace 'user_id' and 'business_id' with the actual IDs from the review data, and DO NOT output any other text or explanation):
        1. <user_id>[TOW]<Date Night>
        2. <business_id>[TOW]<Romantic Ambiance>
        3. <user_id>[TOW]<Good Value>
        4. <business_id>[TOW]<Italian Cuisine>
        5. <user_id>[TOW]<Attentive Service>
        6. <business_id>[TOW]<Outdoor Seating>""",

        'office': """The following are the reviews of a user about a specific office products. Please extract all possible causal hidden factors or attributes related to the 
        reason for the purchase of these products and other products he may need.

Consider the following potential hidden factors:
1. Product Type(e.g. ink, printer)
2. other tools may be needed (e.g., printer, paper, ink cartridges)
3. Product variety (e.g., color options, size options, different styles)
4. Ease of use (e.g., user-friendly design, ergonomic features, comfort)
5. Product quality (e.g., durability, reliability, performance over time)
6. Price sensitivity (e.g., cost-effectiveness, discounts, promotions)
7. Packaging (e.g., packaging size, convenience, eco-friendly)
8. Product customization (e.g., personalized features, adjustable settings)
9. Compatibility (e.g., works with other office supplies, devices, or surfaces)
10. Convenience (e.g., ease of storage, portability, compact design)
11. Non-toxic materials (e.g., safe for children, eco-friendly materials)
12. Specialty features (e.g., waterproof, fade-resistant, extra functionalities)
13. User needs (e.g., for occasional use, frequent use, special events)
14. Multi-functionality (e.g., items that serve more than one purpose)
15. Gift potential (e.g., suitable for gifting, appealing to specific audiences)
16. Social proof (e.g., positive reviews, high ratings, recommendations from others)
17. Brand reputation (e.g., well-known brands like BIC, Avery, Canon)
18. Usage scenario (e.g., home office, school, business use, personal use)

If the user frequently buys a new printer, there might be edges like <user>[TOW]<ink> and <item>[TOW]<printer>, he may need to buy ink cartridges, so we could recommend him with other good ink cartridges.
For the user reviews provided below, please extract all the hidden factors that align with the aforementioned examples.

Here are the user reviews:
{reviews}

Please output the hidden factors in the following structured format:
1. <user_id>[TOW]<other products may be needed>
2. <item_id>[TOW]<Brand Preference>
3. <user_id>[TOW]<Product type>
4. <item_id>[TOW]<Ease of Use>
5. <user_id>[TOW]<Convenience>
6. <item_id>[TOW]<Non-toxic Materials>
7. <user_id>[TOW]<Multi-functionality>
8. <item_id>[TOW]<Gift Potential>

AND DO NOT REPLY ANY OTHER INFORMATION."""
    }

    # Get prompts for the dataset
    system_prompt = system_prompts.get(dataset_name, "")
    user_prompt_template = user_prompt_templates.get(dataset_name, "")
    
    # Format prompts for OpenAI API
    formatted_prompts = []
    for review in reviews:
        user_prompt = user_prompt_template.format(reviews=review)
        # Convert to message format for OpenAI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        formatted_prompts.append(messages)
    
    return formatted_prompts

# Function to make API requests with retries and rate limiting
def make_api_request(client, messages, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.6,
                max_tokens=512,
                top_p=0.9,
                presence_penalty=1.0,
                frequency_penalty=0.5,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                # Apply exponential backoff
                sleep_time = retry_delay * (2 ** attempt)
                print(f"API request failed. Retrying in {sleep_time} seconds... Error: {str(e)}")
                time.sleep(sleep_time)
            else:
                print(f"API request failed after {max_retries} attempts. Error: {str(e)}")
                return ""

# Process a batch of reviews in parallel
def process_batch(batch_data):
    client = get_openai_client()
    dataset_name, reviews = batch_data
    
    prompts = get_prompt(dataset_name, reviews)
    results = []
    
    # Process each review in the batch
    for i, messages in enumerate(prompts):
        response_text = make_api_request(client, messages)
        
        # Parse the response to extract hidden factors
        pattern = r'<([^>]+)>\[TOW\]<([^>]+)>'
        hidden_factors = re.findall(pattern, response_text)
        
        # Parse the current review's JSON data to get real IDs
        review_data = json.loads(reviews[i])
        user_id = review_data['user_id']
        # pdb.set_trace()
        
        if dataset_name == 'CDs' or dataset_name == 'office':
            item_id = review_data['items']['item_id']
        elif dataset_name == 'Movies':
            item_id = review_data['movie_id']
        
        # Process each factor (limit to max 10 factors per review)
        max_factors = 10
        for factor in hidden_factors[:max_factors]:
            entity, attribute = factor
            
            # Replace placeholder entity names with actual IDs
            if entity in ['<user_id>', 'user_id', 'user']:
                entity = user_id
            elif entity in ['<movie_id>', 'movie_id', 'movie']:
                entity = item_id
            elif entity in ['<item_id>', 'item_id', 'item']:
                entity = item_id
            
            results.append([entity, attribute])
    
    return results

# Main function to extract hidden factors with concurrent processing
def extract_hidden_factors_concurrent(all_reviews, dataset_name='CDs', batch_size=4, max_workers=5):
    # Split all reviews into smaller batches for concurrent processing
    batches = []
    for i in range(0, len(all_reviews), batch_size):
        batch = all_reviews[i:i+batch_size]
        batches.append((dataset_name, batch))
    
    all_results = []
    
    # Process batches concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches and collect futures
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                          total=len(batches), 
                          desc="Processing batches"):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Batch processing failed: {str(e)}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden factors from reviews")
    parser.add_argument('--dataset', type=str, default='office', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=10, help='Individual batch size for each API request')
    parser.add_argument('--max_workers', type=int, default=60, help='Maximum number of concurrent workers')
    args = parser.parse_args()

    dataset = args.dataset
    review_path = f'../data/{dataset}/full_dataset.jsonl'
    output_file = f'../data/{dataset}/llm_extracted_data.json'
    
    # Clear output file
    with open(output_file, 'w') as file:
        file.write('')
    
    # Count total lines
    with open(review_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    sample_size = int(total_lines * 1)
    selected_lines = random.sample(range(total_lines), sample_size) if sample_size < total_lines else range(total_lines)
    selected_lines_set = set(selected_lines)

    # Read all reviews into memory
    all_reviews = []
    with open(review_path, 'r', encoding="utf-8") as f:
        print("Reading reviews...")
        for id, line in enumerate(tqdm(f, total=total_lines, desc="Loading Reviews", unit="lines")):
            if id in selected_lines_set:
                all_reviews.append(line)
    
    # Process all reviews in batches concurrently
    print(f"Processing {len(all_reviews)} reviews with {args.max_workers} workers...")
    
    # Process reviews in chunks to avoid memory issues
    chunk_size = 300  # Process 100 reviews at a time
    for i in range(0, len(all_reviews), chunk_size):
        chunk = all_reviews[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(all_reviews) + chunk_size - 1)//chunk_size}")
        
        hidden_factors = extract_hidden_factors_concurrent(
            chunk,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            max_workers=args.max_workers
        )
        
        # Write results to file
        with open(output_file, 'a', encoding="utf-8") as output_f:
            for factor in hidden_factors:
                output_f.write(f"{factor}\n")
        
        print(f"Chunk complete. Total factors extracted: {len(hidden_factors)}")