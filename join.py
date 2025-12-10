import json
import os

def process_reddit_data(file_pairs, output_path):
    """
    Processes pairs of Reddit submission and comment files and combines them
    into a single, nested JSON file.

    The output structure is a dictionary where each key is a submission ID,
    and the value is the submission object with an added 'comments' list.
    
    Args:
        file_pairs (list): A list of tuples, where each tuple is
                           (path_to_submissions_file, path_to_comments_file).
        output_path (str): The file path to save the final combined JSON.
    """
    
    submissions_data = {}
    
    # --- Step 1: Process all submissions first ---
    # This builds our main dictionary of all posts.
    
    print("--- Step 1: Processing all submissions... ---")
    total_submissions = 0
    
    for sub_file, _ in file_pairs:
        print(f"Reading submissions from: {sub_file}")
        try:
            with open(sub_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        submission = json.loads(line)
                        sub_id = submission.get('id')
                        
                        if not sub_id:
                            print(f"Warning: Found submission with no ID in {sub_file}. Skipping.")
                            continue
                            
                        # Add the submission to our main dictionary if it's not already there
                        if sub_id not in submissions_data:
                            submission['comments'] = []  # Add the new key for comments
                            submissions_data[sub_id] = submission
                            total_submissions += 1
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line in {sub_file}: {line.strip()}")
                        
        except FileNotFoundError:
            print(f"Error: File not found {sub_file}. Skipping.")
        except Exception as e:
            print(f"Error reading {sub_file}: {e}")
            
    print(f"Processed {total_submissions} unique submissions in total.")
    
    # --- Step 2: Process all comments and link them ---
    # Now we read the comments and append them to the 'comments' list
    # in our 'submissions_data' dictionary.
    
    print("\n--- Step 2: Processing and linking all comments... ---")
    total_comments_linked = 0
    total_comments_orphaned = 0
    
    for _, comm_file in file_pairs:
        print(f"Reading comments from: {comm_file}")
        try:
            with open(comm_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        comment = json.loads(line)
                        link_id_full = comment.get('link_id')
                        
                        if not link_id_full:
                            print(f"Warning: Found comment with no link_id in {comm_file}. Skipping.")
                            continue
                        
                        # The link_id in comments has a prefix (e.g., "t3_f7ia3")
                        # We need to extract the ID part ("f7ia3")
                        link_id = link_id_full.split('_')[-1]
                        
                        # Find the parent submission and append the comment
                        if link_id in submissions_data:
                            submissions_data[link_id]['comments'].append(comment)
                            total_comments_linked += 1
                        else:
                            # This comment's parent post isn't in our submission files
                            total_comments_orphaned += 1
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line in {comm_file}: {line.strip()}")
                        
        except FileNotFoundError:
            print(f"Error: File not found {comm_file}. Skipping.")
        except Exception as e:
            print(f"Error reading {comm_file}: {e}")
            
    print(f"Successfully processed and linked {total_comments_linked} comments.")
    print(f"Could not find parent submission for {total_comments_orphaned} comments (orphaned).")

    # --- Step 3: Save the combined data to a single file ---
    
    print(f"\n--- Step 3: Saving combined data to {output_path}... ---")
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Using indent=2 makes the file human-readable, but creates a larger file.
            # Remove 'indent=2' for a compact, single-line file.
            json.dump(submissions_data, f, indent=2)
            
        print(f"Successfully saved combined data!")
        print(f"Output file location: {output_path}")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not save output file: {e}")


if __name__ == "__main__":
    # Define the base directory
    base_dir = "/Users/igor/Downloads/reddit/subreddits24"
    
    # List of subreddit names to process
    subreddits = [
        "MushroomSupplements",
        "NooTopics",
        "NootropicsDepot",
        "Preworkoutsupplements",
        "Supplements"
    ]
    
    # Generate the file pairs
    file_pairs_to_process = []
    for sub in subreddits:
        sub_file = os.path.join(base_dir, f"{sub}_submissions")
        comm_file = os.path.join(base_dir, f"{sub}_comments")
        file_pairs_to_process.append((sub_file, comm_file))
        
    # Define the output file path
    # We'll save it one level up from the 'subreddits24' folder.
    output_dir = os.path.dirname(base_dir)
    output_json_path = os.path.join(output_dir, "combined_reddit_data.json")
    
    # Run the processing function
    process_reddit_data(file_pairs_to_process, output_json_path)
