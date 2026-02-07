import js from "@eslint/js";

export default [
    js.configs.recommended,
    {
        files: ["**/*.js"],
        ignores: ["eslint.config.js"],
        languageOptions: {
            ecmaVersion: 2022,
            sourceType: "script",
            globals: {
                document: "readonly",
                window: "readonly",
                console: "readonly",
                fetch: "readonly",
                URL: "readonly",
                Blob: "readonly",
                FormData: "readonly",
                TextDecoder: "readonly",
                AbortController: "readonly",
                EventSource: "readonly",
                setTimeout: "readonly",
            },
        },
        rules: {
            "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
            "no-console": "off",
            "prefer-const": "error",
            "no-var": "error",
            eqeqeq: ["error", "always"],
            curly: ["error", "all"],
        },
    },
];
