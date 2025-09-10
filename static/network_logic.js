document.addEventListener("DOMContentLoaded", () => {
    const cyContainer = document.getElementById('cy');
    const deviceSelector = document.getElementById('device-select');
    const networkTypeSelector = document.getElementById('network-type-select');
    const weightSlider = document.getElementById('weight-threshold');
    const thresholdLabel = document.getElementById('threshold-value');
    const infoBox = document.getElementById('info-box');

    let cy = null;
    let selectedNodes = [];
    let currentNetworkType = networkTypeSelector.value;

    // === TABLE INIT ===
    const dataTable = $('#summary-table').DataTable({
        paging: false,
        searching: false,
        info: false,
        ordering: true,
        columns: [
            { title: "Node" },
            { title: "AvgAbs" },
            { title: "Total" },
            { title: "AvgRel" },
            { title: "Flow" }
        ]
    });

    // === GRAPH LOAD ===
    function loadGraph(device) {
        fetch(`/graph?device=${device}`)
            .then(res => res.json())
            .then(data => {
                cy = cytoscape({
                    container: cyContainer,
                    elements: data.elements,
                    layout: { name: 'preset' },
                    style: getGraphStyle(),
                    selectionType: 'additive',
                    boxSelectionEnabled: true
                });

                initializeEventHandlers();
                setWeightThresholdSlider();
                applyWeightThreshold(0);
            });
    }

    // === STYLE SETUP ===
    function getGraphStyle() {
        return [
            {
                selector: 'node',
                style: {
                    'label': 'data(id)',
                    'background-color': 'rgba(0, 216, 255, 1)',
                    'color': 'rgba(255, 255, 255, 1)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '4px',
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'selectable': true
                }
            },
            {
                selector: 'edge[group = "domain"]',
                style: {
                    'line-color': 'data(color)',
                    'width': 'data(width)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': 'blue',
                    'arrow-scale': 0.2,
                    'target-arrow-fill': 'hollow'
                }
            },
            {
                selector: 'edge[group = "network"]',
                style: {
                    'line-color': 'data(color)',
                    'width': 'data(width)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': 'red',
                    'target-arrow-fill': 'hollow',
                    'arrow-scale': 0.2
                }
            },
            {
                selector: 'edge.highlighted',
                style: {
                    'line-color': 'rgba(0, 255, 0, 1)',
                    'width': 4,
                    'opacity': 1.0,
                    'z-index': 9999
                }
            },
            {
            selector: 'node.hovered',
            style: {
                'border-width': 6,
                'border-color': 'rgba(52, 115, 70, 1)'
            }
            },
            {
            selector: 'node:selected',
            style: {
                'border-width': 6,
                'border-color': 'rgba(237, 203, 6, 0.6)'
            }
            },
            {
                selector: 'core',
                style: { 'background-color': '#000000' }
            }
        ];
    }

    // control edge visibility based on weight threshold    
    function applyWeightThreshold(threshold) {
        if (!cy) return;
        cy.edges().forEach(edge => {
            edge.style('display', edge.data('weight') >= threshold ? 'element' : 'none');
        });
    }

    // set slider max based on max edge weight
    function setWeightThresholdSlider() {
        let maxWeight = 0;
        cy.edges().forEach(edge => {
            const w = edge.data('weight');
            if (w > maxWeight) maxWeight = w;
        });

        weightSlider.max = Math.ceil(maxWeight);
        weightSlider.value = 0;
        thresholdLabel.textContent = "0";
    }

    // update summary table with selected nodes
    function updateSummaryTable(nodes) {
        dataTable.clear();

        nodes.forEach(node => {
            const d = node.data();
            dataTable.row.add([
                d.id,
                d.AvgAbs?.toFixed(2) || 'n/a',
                d.Total?.toFixed(2) || 'n/a',
                d.AvgRel?.toFixed(2) || 'n/a',
                d.flow?.toFixed(2) || 'n/a'
            ]);
        });

        dataTable.draw();
    }

    // === CY EVENT HANDLERS ===
    function initializeEventHandlers() {

        // Node hover effect
        cy.nodes().on('mouseover', evt => {
            evt.target.addClass('hovered');
            showNodeInfo(evt.target);
        });

        cy.nodes().on('mouseout', evt => {
            evt.target.removeClass('hovered');
            hideInfoBox();
        });

        // Edge hover → info panel
        cy.edges().on('mouseover', evt => showEdgeInfo(evt.target));
        cy.edges().on('mouseout', () => hideInfoBox());

        // Shift+click = shortest path trigger
        cy.nodes().on('tap', evt => {
            const node = evt.target;
            const id = node.id();
            const isShift = evt.originalEvent.shiftKey;

            if (!isShift) return;

            if (selectedNodes.includes(id)) {
                selectedNodes = selectedNodes.filter(n => n !== id);
                node.unselect();  // ✅ fix here
            } else {
                selectedNodes.push(id);
                node.select();    // ✅ uses :selected style
            }

            if (selectedNodes.length === 2) {
                const [src, dst] = selectedNodes;
                selectedNodes.forEach(n => cy.getElementById(n).unselect());  // ✅ fix here
                selectedNodes = [];
                getShortestPaths(src, dst);
            }
        });


        // Box selection → update table
        cy.on('select', 'node', () => {
            updateSummaryTable(cy.nodes(':selected'));
        });

        cy.on('unselect', 'node', () => {
            updateSummaryTable(cy.nodes(':selected'));
        });
    }

    function getShortestPaths(source, target) {
        const params = new URLSearchParams({
            src: source,
            dst: target,
            device: deviceSelector.value,
            max_paths: 5
        });

        fetch(`/shortest-path?${params.toString()}`)
            .then(res => res.json())
            .then(data => {
                cy.edges().removeClass('highlighted');
                if (!data.paths || data.paths.length === 0) {
                    alert(data.message || "No paths found.");
                    return;
                }

                const allPathEdges = new Set();
                data.paths.forEach(p => {
                    for (let i = 0; i < p.path.length - 1; i++) {
                        const u = p.path[i];
                        const v = p.path[i + 1];
                        const edge = cy.edges().filter(e => e.source().id() === u && e.target().id() === v);
                        edge.forEach(e => allPathEdges.add(e));
                    }
                });
                allPathEdges.forEach(e => e.addClass('highlighted'));

                infoBox.innerHTML = `
                    <strong>${data.paths.length} shortest paths from ${source} → ${target}</strong><br><br>
                    ${data.paths.map((p, i) =>
                        `<strong>Path ${i + 1}:</strong> ${p.path.join(' → ')}<br>Score: ${p.score_abs.toFixed(2)}`
                    ).join('<br><br>')}
                `;
                infoBox.style.display = 'block';
            })
            .catch(err => {
                console.error("Error fetching shortest path:", err);
                alert("Failed to fetch shortest path.");
            });
    }

    function showEdgeInfo(edge) {
        const data = edge.data();
        infoBox.innerHTML = `
            <strong>Edge: ${data.source} → ${data.target}</strong><br>
            Weight: ${data.weight || 'n/a'}
        `;
        infoBox.style.display = 'block';
    }

    function showNodeInfo(node) {
        const data = node.data();
        infoBox.innerHTML = `
            <strong>Node: ${data.id}</strong><br>
            <br>    
            AvgAbs: ${data.AvgAbs?.toFixed(2) || 'n/a'}s<br>
            AvgRel: ${data.AvgRel?.toFixed(2) || 'n/a'}s<br>
            Total: ${data.Total?.toFixed(2) || 'n/a'}s<br>
            <br>
            Flow: ${data.flow?.toFixed(2) || 'n/a'}<br> 
        `;
        infoBox.style.display = 'block';
    }

    function hideInfoBox() {
        infoBox.style.display = 'none';
    }

    // === CONTROL EVENTS ===
    deviceSelector.addEventListener('change', () => loadGraph(deviceSelector.value));
    networkTypeSelector.addEventListener('change', () => {
        currentNetworkType = networkTypeSelector.value;
        applyNetworkStyle();
    });

    weightSlider.addEventListener('input', () => {
        const threshold = parseInt(weightSlider.value);
        thresholdLabel.textContent = threshold;
        applyWeightThreshold(threshold);
    });

    document.getElementById('clear-selection').addEventListener('click', () => {
        cy.nodes().unselect();
        cy.edges().removeClass('highlighted');
        infoBox.style.display = 'none';
        dataTable.clear().draw();
    });

    // === INITIALIZE ===
    loadGraph(deviceSelector.value);
});
