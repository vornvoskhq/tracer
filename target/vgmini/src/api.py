from flask import Flask, jsonify
import os

app = Flask(__name__)

CONFIG_DIR = "results/configs"

@app.route('/api/configs', methods=['GET'])
def get_configs():
    """
    Returns a list of available configuration files.
    """
    if not os.path.exists(CONFIG_DIR):
        return jsonify({"error": "Configuration directory not found"}), 404

    configs = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
    return jsonify(configs)

if __name__ == '__main__':
    app.run(debug=True)
