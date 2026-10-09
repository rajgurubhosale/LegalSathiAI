export default function ({ data, setStateValue }) {
    const storageKey = "legalsaathi.access_token";
    let result;
    try {
        if (data.action === "clear") {
            sessionStorage.removeItem(storageKey);
        } else if (data.action === "save") {
            sessionStorage.setItem(storageKey, data.token);
        }
        result = {
            token: sessionStorage.getItem(storageKey),
            action: data.action,
            available: true,
        };
    } catch {
        result = { token: null, action: data.action, available: false };
    }
    if (JSON.stringify(result) !== JSON.stringify(data.previous)) {
        setStateValue("result", result);
    }
}
