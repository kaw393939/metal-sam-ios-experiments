import os

def export_codebase(root_dir, output_file):
    # Extensions to include
    extensions = {'.swift', '.metal', '.h', '.c', '.cpp', '.hpp', '.py', '.md'}
    # Directories to exclude
    exclude_dirs = {'.build', '.swiftpm', '.git', '.agent', 'artifacts', 'runs'}
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk source directory
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Prune excluded dirs
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            
            for filename in filenames:
                ext = os.path.splitext(filename)[1]
                if ext in extensions:
                    filepath = os.path.join(dirpath, filename)
                    relpath = os.path.relpath(filepath, root_dir)
                    
                    outfile.write(f"\n\n// --- FILE: {relpath} ---\n\n")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"// Error reading file: {e}\n")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = os.path.join(project_root, "ordo_sam3_sensor_export.txt")
    
    print(f"Exporting codebase from {project_root} to {output_path}...")
    export_codebase(project_root, output_path)
    print("Done.")
