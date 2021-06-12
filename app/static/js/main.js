function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

const req = (filename) => {
    fetch(`/process/${filename}`, {
        method: "POST"
    }).then(() => {
        status(false);
    })
}

const status = (isPolling) => {
    fetch("/status/", {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }).then(
        res => res.json()
    ).then(res => updateEntry(res, isPolling));
}

const updateEntry = async (s, isPolling) => {
    let table = document.getElementById("table")
    table.innerText = "";
    for (const [key, value] of Object.entries(s)) {
        let e = document.createElement("div");
        const t = document.createTextNode(key);
        e.className = "entry"
        if (!value.includes("processing") && !value.includes("processed")) {
            let b = document.createElement("button");
            let s = document.createElement("span");
            b.onclick = () => req(key);
            s.className = "material-icons"; s.innerText = "cloud_upload";
            b.appendChild(s); e.appendChild(b);
        } else if (value.includes("processing")) {
            let s = document.createElement("span");
            s.className = "material-icons"; s.innerText = "sync";
            e.appendChild(s);
        } else if (value.includes("processed")) {
            let a = document.createElement("a");
            let s = document.createElement("span");
            s.className = "material-icons"; s.innerText = "done";
            a.href = `/download/processed/${key}`;
            a.appendChild(s); e.appendChild(a);
        }
        e.appendChild(t);
        table.appendChild(e);
    }

    if(isPolling) {
        await sleep(1000);
        status(isPolling);
    }
}