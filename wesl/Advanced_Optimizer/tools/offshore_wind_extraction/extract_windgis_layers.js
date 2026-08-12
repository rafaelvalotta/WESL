/*
 * Extracts layers from the Sea Impact / WindGIS map (Mapbox GL) as GeoJSON,
 * directly from vectors already rendered by the browser - no manual tracing/drawing required.
 *
 * HOW TO USE - see TUTORIAL.md in this same folder for the full walkthrough.
 * Short version:
 * 1. Open https://sea-impact.com/offshore-wind-map/ and let the map load.
 * 2. The actual map runs inside an iframe at viewer.eu.windgis.lautec.com.
 *    Open this iframe directly in a tab (get the iframe "src" via
 *    devtools -> Elements, or check the network tab) so you can run the console in it.
 * 3. In the layer selector (layers icon), turn on the layers you want to capture
 *    (Wind Turbine Generators, Floating WTG, Substations & Converters,
 *    Inter-Array Cables, Export Cables, Mooring Lines, Wind Farms).
 * 4. Paste this whole file into the DevTools Console (F12). It finds the
 *    mapboxgl.Map instance and stores it at window.__map.
 * 5. Position the map (window.__map.fitBounds([[minLon,minLat],[maxLon,maxLat]])),
 *    wait a few seconds for tiles to settle, then call extractAll() (defined
 *    at the end of this script) to pull every visible layer into
 *    window.__extracted, or use querySourceFeatures directly against the
 *    known source/sourceLayer pairs printed to the console (stable across the
 *    whole global dataset - only the map view changes per region).
 * 6. Copy JSON.stringify(window.__extracted), or call downloadExtracted() to
 *    save it straight to a .json file (safer for large regions).
 *
 * IMPORTANT - TILE CLIPPING AT EDGES
 * Vector tiles (Mapbox) cut polygons and lines exactly at the edge of each
 * tile. If the cluster you want spans more than 1 tile at the current zoom level, each
 * polygon will appear FRAGMENTED into several pieces sharing the SAME properties
 * (e.g., same wf_name). This is expected, and resolved by dissolving (union) the
 * fragments that share the same key property - use build_region_pipeline.py
 * (shapely.ops.unary_union) to reconstruct the single, continuous polygon for
 * each lease area. Points (turbines, substations) only duplicate if they fall
 * exactly on a tile edge - deduping by rounded coordinates solves it.
 *
 * IMPORTANT - CABLES NEED HIGHER ZOOM THAN BOUNDARIES/TURBINES
 * A single wide, low-zoom pass (~6-7) reliably captures lease-area polygons
 * and turbine/substation points for an entire region. Inter-array/export
 * cable LINES come back empty at that same low zoom - the tileset appears to
 * drop thin line features at low zoom. Capturing cables needs zoom ~8-9 per
 * sub-area (several passes across a large region). Budget extra time if you
 * need them.
 */

(function () {
  // 1) locate the mapboxgl.Map instance stored in React's state
  function findMapInstance() {
    const canvas = document.querySelector('.mapboxgl-canvas');
    if (!canvas) throw new Error('Mapbox canvas not found on this page.');
    let el = canvas;
    let fiberHost = null;
    while (el) {
      const key = Object.keys(el).find(
        (k) => k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$')
      );
      if (key) {
        fiberHost = { el, key };
        break;
      }
      el = el.parentElement;
    }
    if (!fiberHost) throw new Error('React Fiber not found starting from canvas.');

    let node = fiberHost.el[fiberHost.key];
    const seen = new Set();
    function scan(obj, depth) {
      if (!obj || typeof obj !== 'object' || depth > 3 || seen.has(obj)) return null;
      seen.add(obj);
      if (typeof obj.getStyle === 'function' && typeof obj.getCanvas === 'function') return obj;
      for (const k of Object.keys(obj)) {
        try {
          const v = obj[k];
          if (v && typeof v === 'object') {
            const r = scan(v, depth + 1);
            if (r) return r;
          }
        } catch (e) {}
      }
      return null;
    }

    let depth = 0;
    while (node && depth < 40) {
      if (node.memoizedState) {
        const r = scan(node.memoizedState, 0);
        if (r) return r;
      }
      if (node.memoizedProps) {
        const r = scan(node.memoizedProps, 0);
        if (r) return r;
      }
      node = node.return;
      depth++;
    }
    throw new Error('Map instance not found in React tree.');
  }

  const map = findMapInstance();
  window.__map = map;

  // 2) list all vector sources that are not the basemap (composite/mapbox-dem)
  const style = map.getStyle();
  const vectorLayers = style.layers.filter(
    (l) => l.source && l['source-layer'] && l.source !== 'composite'
  );

  console.log('Data sources found (source/sourceLayer pairs are stable across regions):');
  const seen = new Set();
  for (const l of vectorLayers) {
    const key = l.source + ' | ' + l['source-layer'];
    if (seen.has(key)) continue;
    seen.add(key);
    console.log(' -', l.id, '(' + l.type + ')', '<-', key);
  }

  // 3) extract currently loaded features (visible in viewport / cached tiles)
  window.extractAll = function extractAll() {
    const extracted = {};
    for (const l of vectorLayers) {
      const source = l.source;
      const sourceLayer = l['source-layer'];
      const feats = map.querySourceFeatures(source, { sourceLayer });
      if (feats.length === 0) continue;
      const key = l.id;
      extracted[key] = {
        type: 'FeatureCollection',
        meta: { source, sourceLayer, layerType: l.type },
        features: feats.map((f) => ({
          type: 'Feature',
          properties: f.properties,
          geometry: f.geometry,
        })),
      };
    }
    window.__extracted = extracted;
    console.log(
      'Extracted to window.__extracted:',
      Object.fromEntries(Object.entries(extracted).map(([k, v]) => [k, v.features.length]))
    );
    return extracted;
  };

  window.downloadExtracted = function downloadExtracted() {
    const blob = new Blob([JSON.stringify(window.__extracted)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'raw_extracted.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  console.log('Ready. Position the map (map.fitBounds(...)), wait a few seconds, then run extractAll() and downloadExtracted().');
})();
