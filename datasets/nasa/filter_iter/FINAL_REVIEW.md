# NASA Caption Pipeline — Production Output Review

**Model**: Qwen3.6-27B (DP=2 TP=2, thinking ON) · **Samples**: 76 · **Captions produced**: 76/76

**Pipeline**:
1. `clean_caption.py` — strips seed boilerplate (HTML, leading archive IDs, inline attributions)
2. `prompts/v3.txt` — filter (KEEP/DROP)
3. `regen_prompts/balanced.txt` — fused rewrite (KEEP branch) or image-only caption (DROP branch)
4. `caption_scrubber.py` — deterministic residual cleanup for meta-refs and NASA-prefix

**Filter accuracy vs labels**: 74/76 (97.4%)
- KEEP branch (filter said KEEP): 44 samples
- DROP branch (filter said DROP, seed ignored, captioned from pixels): 32 samples

**Samples where scrubber modified output**: 4

---

## KEEP branch — seed context fused with image  _(44)_

### 0600583  

<img src="images_thumb/0600583.jpg" width="420"/>

**Seed (cleaned):**
> At its founding, the Marshall Space Flight Center (MSFC) inherited the Army's Jupiter and Redstone test stands, but much larger facilities were needed for the giant stages of the Saturn V. From 1960 to 1964, the existing stands were remodeled and a sizable new test area was developed. Construction of the S-IC Static test stand complex began in 1961 in the west test area of MSFC, and was completed in 1964. The S-IC static test stand was designed to develop and test the 138-ft long and 33-ft diameter Saturn V S-IC first stage, or booster stage, weighing in at 280,000 pounds. Required to hold down the brute force of a 7,500,000-pound thrust produced by 5 F-1 engines, the S-IC static test stand was designed and constructed with the strength of hundreds of tons of steel and 12,000,000 pounds of cement, planted down to bedrock 40 feet below ground level.

**Final caption** (590 chars):
> Massive concrete towers, topped with metal scaffolding and frameworks, dominate the construction site as tall cranes loom overhead. A vintage pickup truck and a sedan are parked in the foreground near a large, curved pipe lying on the ground. These structures are part of the S-IC Static test stand complex at Marshall Space Flight Center, built to support and test the Saturn V S-IC first stage. The facility was engineered with 12 million pounds of cement and hundreds of tons of steel to anchor the booster and withstand the 7.5 million pounds of thrust produced by its five F-1 engines.

---

### 7022489  

<img src="images_thumb/7022489.jpg" width="420"/>

**Seed (cleaned):**
> This photograph of an astronaut getting the Lunar Roving Vehicle (LRV) ready for exploration of the lunar surface was taken during activities of the Apollo 15 mission. Designed and developed by the Marshall Space Flight Center and built by the Boeing Company, the LRV was first used on the Apollo 15 mission and increased the range of astronauts' mobility and productivity on the lunar surface.

**Final caption** (661 chars):
> The Lunar Roving Vehicle (LRV) sits on the gray lunar surface beside the gold-foil-wrapped descent stage of the Lunar Module, its tubular frame supporting wire-mesh wheels, a high-gain antenna mast, and folded crew seats. An astronaut in a white spacesuit stands near the rover, appearing to prepare equipment, while deep tire tracks and footprints mark the foreground regolith. A "UNITED STATES" flag patch is visible on the side of the Lunar Module. The LRV was designed and developed by Marshall Space Flight Center and built by the Boeing Company, first used on the Apollo 15 mission to extend the range and productivity of crewed lunar-surface exploration.

---

### AFRC2019-0290-39  

<img src="images_thumb/AFRC2019-0290-39.jpg" width="420"/>

**Seed (cleaned):**
> The TigerShark unmanned aircraft by Navmar Applied Sciences Corporation flew over the skies at NASA's Armstrong Flight Research Center for the Unmanned Aircraft Systems integration in the National Airspace Systems, Flight Test Series Six (FT6) project. FT6 focused on low size weight and power sensors for Detect and Avoid (DAA) operations in controlled airspace to inform the FAA through the RTCA Special Committee DAA Working Group on the phase 2 minimum operational performance standards for DAA and air-to-air radar.

**Final caption** (582 chars):
> The TigerShark unmanned aircraft, built by Navmar Applied Sciences Corporation, flies through a clear blue sky with its tricycle landing gear deployed. The white drone features high-mounted wings with bright orange tips and a pusher propeller at the rear. It is participating in the Unmanned Aircraft Systems integration in the National Airspace Systems, Flight Test Series Six (FT6) project. This mission focuses on testing low size, weight, and power sensors for Detect and Avoid (DAA) operations in controlled airspace to help establish minimum operational performance standards.

---

### ARC-1968-A-41374  · scrubber applied

<img src="images_thumb/ARC-1968-A-41374.jpg" width="420"/>

**Seed (cleaned):**
> 2-Prop. R.C.F. (Rotating Cylinder Flap) in 40 x 80ft. wind tunnel. rear view detail of flap

**Final caption** (510 chars):
> A large, polished metallic cylinder runs horizontally across the frame, its surface marked by faint panel lines and rivets. Below the cylinder, thin metal rods curve downward to connect to a rectangular mounting plate secured to a white vertical surface. This setup is a Rotating Cylinder Flap (RCF) test article, specifically a two-propeller configuration, positioned inside a 40 x 80-foot wind tunnel. The rear detail of the flap mechanism, showing the linkage and the smooth, curved surface of the cylinder.

<details><summary>Pre-scrubber output</summary>

> A large, polished metallic cylinder runs horizontally across the frame, its surface marked by faint panel lines and rivets. Below the cylinder, thin metal rods curve downward to connect to a rectangular mounting plate secured to a white vertical surface. This setup is a Rotating Cylinder Flap (RCF) test article, specifically a two-propeller configuration, positioned inside a 40 x 80-foot wind tunnel. The view captures the rear detail of the flap mechanism, showing the linkage and the smooth, curved surface of the cylinder.

</details>

---

### ARC-1979-A79-7028  

<img src="images_thumb/ARC-1979-A79-7028.jpg" width="420"/>

**Seed (cleaned):**
> These four pictures of Jupiter's Great Red Spot were taken Feb. 2 and 3, 1979, when Voyager 1 was about 31 million kilometers (19.4 million miles) from Jupiter. The pictures were taken one Jupiter rotation apart, so that together they depict four days in the life of the centuries-old Red Spot. The pictures clearly demonstrate changes in circulation around the Red Spot during the 40-hour period. The photos were taken through a blue filter.

**Final caption** (429 chars):
> Jupiter's Great Red Spot appears as a large, dark, oval-shaped storm in four black-and-white panels, surrounded by swirling bands of lighter and darker clouds. Text below the images reads "GREAT RED SPOT - ROTATIONS 67-70" and "FEBRUARY 2 THROUGH FEBRUARY 3." Imaged by the Voyager 1 spacecraft, the sequence demonstrates changes in the atmospheric circulation surrounding the centuries-old storm over a period of four rotations.

---

### ARC-1981-AC81-7065  · scrubber applied

<img src="images_thumb/ARC-1981-AC81-7065.jpg" width="420"/>

**Seed (cleaned):**
> Range : 2.3 million km. ( 1.4 million miles ) P-24067C This Voyager 2 photograph of Titan, a satellite of Saturn, shows some detail in the cloud systems. The southern hemisphere appears lighter in contrast, a well defined band is seen near the equator, and a dark collar is evident at the north pole. All these bands are associated with the cloud circulation in titan's atmosphere. The extended haze, composed of of sub-micron size particles, is seen clearly around the satellite's limb. This image was composed from blue, green, and violet frames.

**Final caption** (467 chars):
> Titan, a satellite of Saturn, appears as a hazy, pale orange sphere against a dark background. A dark collar is evident at the north pole, a well-defined band is visible near the equator, and the southern hemisphere appears lighter in contrast. These distinct bands are associated with cloud circulation in Titan's atmosphere. An extended haze, composed of sub-micron size particles, is clearly visible around the satellite's limb. Imaged by the Voyager 2 spacecraft.

<details><summary>Pre-scrubber output</summary>

> Titan, a satellite of Saturn, appears as a hazy, pale orange sphere against a dark background. A dark collar is evident at the north pole, a well-defined band is visible near the equator, and the southern hemisphere appears lighter in contrast. These distinct bands are associated with cloud circulation in Titan's atmosphere. An extended haze, composed of sub-micron size particles, is clearly visible around the satellite's limb. The image was captured by the Voyager 2 spacecraft and composed from blue, green, and violet frames.

</details>

---

### ARC-1990-A90-3000  

<img src="images_thumb/ARC-1990-A90-3000.jpg" width="420"/>

**Seed (cleaned):**
> This Magellan image reveals Sacajawea Patera, a large, elongate caldera located in wester Ishtar Terra on the smooth plateau of Lakshmi Planum. The image is centered at 64.5 degrees north latitude adn 337 degrees east longitude. It is approx. 420 km (252 mi.) wide at the base. Sacajawea is a depression approx. 1-2 km (0.6-1.2 mi.) deep and 120 by 215 km (74 by 133 mi.) in diameter; it is elongate in a sousthwest-northeast direction.

**Final caption** (489 chars):
> Sacajawea Patera, a large, elongate caldera located in Ishtar Terra on the smooth plateau of Lakshmi Planum, is imaged by the Magellan spacecraft as a diagonal strip of granular, grayscale terrain. The frame displays a noisy surface texture with a prominent bright linear feature running from the upper center toward the lower left. The volcanic depression is oriented in a southwest-northeast direction, with subtle variations in brightness suggesting the topography of the caldera floor.

---

### ARC-1994-AC94-0015-1  

<img src="images_thumb/ARC-1994-AC94-0015-1.jpg" width="420"/>

**Seed (cleaned):**
> N-243 VMS N-Cab HELMEE Project. CLOSE-UPS OF INSTRUMENT PANEL & FLIGHT CONTROLS.

**Final caption** (616 chars):
> The cockpit of the N-243 VMS N-Cab, associated with the HELMEE Project, features a dark instrument panel densely populated with analog flight instruments and a central rectangular display screen. Rows of toggle switches and indicator lights are arranged along the top and sides of the dashboard, flanking circular gauges for flight parameters. A control stick with a buttoned grip extends from the center console, positioned near a throttle lever. In the foreground, a teal headset rests near a seat covered with a dark, textured blanket, while the side windows reveal a concrete floor and hangar structures outside.

---

### ARC-1994-AC94-0109-3  

<img src="images_thumb/ARC-1994-AC94-0109-3.jpg" width="420"/>

**Seed (cleaned):**
> ER-2: ASHOE/MAESA Expidition art Airborne Southern Hemisphere Ozone Experiment; Measurements for Assessing the Effects of Stratospheric Aircraft - Growth of Antarctic Ozone hole September - October 1993 (false color data from TOMS staellite, Goddard Space Flight Center)

**Final caption** (744 chars):
> Two polar projection maps illustrate the "Growth of Antarctic Ozone Hole September - October 1993," displaying ozone concentration levels over the continent. The top map, labeled "September 1, 1993," shows a central area of lower ozone concentration in green and blue hues, surrounded by rings of yellow, orange, and red indicating higher concentrations. The bottom map, labeled "October 1, 1993," depicts a significantly larger and deeper ozone hole, characterized by a large central region of dark blue and purple. A color scale on the right of each map indicates "O3 Concentration" in "DOBSON UNITS," ranging from 80 to 400. The visualization is identified at the bottom as "False Color Data from TOMS Satellite Goddard Space Flight Center."

---

### E-5070  

<img src="images_thumb/E-5070.jpg" width="420"/>

**Seed (cleaned):**
> Side view of a F-105B (serial #54-0102) photographed on Rogers Dry Lakebed at Edwards Air Force Base, California in 1959. The black stripes across the left wheel-panel complete the lettering on the bottom of the wing when wheels are retracted. Two of the F-105B characteristics are fuselage length of 61 feet 1.33 inches and a wing area of 385.0 square feet.

**Final caption** (518 chars):
> A silver F-105B fighter jet sits on a flat, light-colored surface under a bright sky. The aircraft features a long, slender nose with a forward probe, a single-seat cockpit canopy, and "U.S. AIR FORCE" lettering along the fuselage with "FH-102" below it. The tail displays the number "40102" and a US Air Force star insignia is visible on the rear fuselage. Black-and-white stripes painted on the left wing panel are designed to complete lettering patterns visible on the wing underside when the landing gear retracts.

---

### EC03-0085-17  

<img src="images_thumb/EC03-0085-17.jpg" width="420"/>

**Seed (cleaned):**
> Scaled Composites' unique tandem-wing Proteus was the testbed for a series of UAV collision-avoidance flight demonstrations. An Amphitech 35GHz radar unit installed below Proteus' nose was the primary sensor for the Detect, See and Avoid tests.

**Final caption** (580 chars):
> The white, tandem-wing Proteus aircraft flies over a vast, sandy desert landscape, its exceptionally long, slender wings spanning the frame. The name "PROTEUS" is printed in black capital letters along the side of the fuselage, while the registration number "N224PG" is visible on the vertical stabilizer. A red-tipped sensor unit, an Amphitech 35GHz radar, is mounted beneath the nose cone. Designed by Scaled Composites, this unique aircraft served as a testbed for UAV collision-avoidance flight demonstrations, utilizing the nose-mounted radar for Detect, See and Avoid tests.

---

### EC05-0030-04  

<img src="images_thumb/EC05-0030-04.jpg" width="420"/>

**Seed (cleaned):**
> A post-flight inspection of the panels on the F-15B's flight test fixture shows five divots of TPS foam were successfully ejected during the LIFT experiment flight #2, the first flight with TPS foam.

**Final caption** (513 chars):
> A large, dark, rectangular test fixture is mounted beneath the wing of a white aircraft, supported by yellow chocks on a concrete tarmac marked with a yellow line. A long, slender metal probe with red and white stripes extends from the front lower corner of the fixture. This is the flight test fixture on an F-15B following the LIFT experiment flight #2, the first flight to utilize TPS foam. A post-flight inspection of the panels confirms that five divots of the foam were successfully ejected during the test.

---

### GRC-1960-C-53766  

<img src="images_thumb/GRC-1960-C-53766.jpg" width="420"/>

**Seed (cleaned):**
> NASA Researchers view a demonstration of the moon dust simulator in the 8- by 6-Foot Supersonic Wind Tunnel facility at the National Aeronautics and Space Administration (NASA) Lewis Research Center. The researchers were studying the effect of the lunar lander's retrorockets on the loose dust on the lunar surface. There was some concern that the retrorockets would kick up so much dust that the crew would lose the ability to see. They also did not know how the dust's behavior would be affected by the space atmosphere. This small vacuum tank was built for very preliminary investigations into this matter.

**Final caption** (687 chars):
> Two researchers look through a large circular window into a test chamber where a small white model of a lunar lander sits on a bed of fine particles. The man on the right, wearing a white shirt and tie, reaches in with a gloved hand near the model, while a cloud of dust swirls around its three landing legs. This setup is a moon dust simulator located in the 8- by 6-Foot Supersonic Wind Tunnel facility, identified by the "NASA 8-6" label on the scale at the bottom of the window. The experiment studies the effect of a lunar lander's retrorockets on loose surface dust, investigating concerns that the exhaust could kick up enough debris to obscure the crew's vision during a landing.

---

### GRC-2008-C-04246  

<img src="images_thumb/GRC-2008-C-04246.jpg" width="420"/>

**Seed (cleaned):**
> Mars Science Laboratory (MSL) Flexible Canopy Testing in the Glenn Research Center, 10x10 Supersonic Wind Tunnel

**Final caption** (461 chars):
> A group of engineers and technicians pose with a large, dark cylindrical test article featuring a smaller attached section and visible wiring in a facility with blue lighting. The object is the Flexible Canopy test article for the Mars Science Laboratory (MSL), designed to test the parachute system for the rover's landing on Mars. The team stands in the 10x10 Supersonic Wind Tunnel at Glenn Research Center, where the component undergoes aerodynamic testing.

---

### GRC-2013-C-04615  

<img src="images_thumb/GRC-2013-C-04615.jpg" width="420"/>

**Seed (cleaned):**
> Advanced Exploration Systems (AES) Modular Power Systems for Space Exploration (AMPS); electrochemistry, AMPS, will infuse and demonstrate batteries, fuel cells, and other power modules for exploration ground system demonstrations

**Final caption** (550 chars):
> A technician in a yellow striped polo shirt leans over a large blue test rig, adjusting a light-colored bar across the top of a grid of electrochemical cells. The apparatus is the Advanced Exploration Systems Modular Power Systems for Space Exploration (AMPS), featuring rows of white modules with black cylindrical connectors and coiled copper wiring. This system is designed to demonstrate batteries, fuel cells, and other power modules for future exploration ground systems. The work is being conducted at the Glenn Research Center at Lewis Field.

---

### GRC-2019-C-03149  

<img src="images_thumb/GRC-2019-C-03149.jpg" width="420"/>

**Seed (cleaned):**
> Advanced Electric Propulsion Systems Contract, Technology Demonstration Unit, TDU-3 Checkout Test Hardware Installed in Vacuum Facility 5, VF-5

**Final caption** (557 chars):
> The Technology Demonstration Unit (TDU-3) for the Advanced Electric Propulsion Systems (AEPS) contract is installed for checkout testing inside a large vacuum chamber. The hardware features a complex framework of copper-colored structural elements and wiring surrounding a large, circular metallic drum with a central hub. A vertical chain mechanism is visible on the right side, while various cables and connectors are routed through blue structural mounts. The background shows the curved, ribbed interior walls of the facility, illuminated by blue light.

---

### GRC-2019-C-09942  

<img src="images_thumb/GRC-2019-C-09942.jpg" width="420"/>

**Seed (cleaned):**
> Virtual Intelligent Planetary Exploration Rover, VIPER Mobility Platform Testing

**Final caption** (560 chars):
> The VIPER mobility platform sits in a large indoor sand pit, its white tubular frame supporting a chassis wrapped in gold thermal foil and large, gold-colored, treaded wheels. A camera mast extends from the top of the rover. To the right, a person wearing a striped shirt, jeans, and a white protective hood holds a thick black tether cable connected to the vehicle, while another individual in similar protective gear sits on the edge of the sand pit to the left. The setup facilitates mobility testing for the Virtual Intelligent Planetary Exploration Rover.

---

### GSFC_20171208_Archive_e001018  

<img src="images_thumb/GSFC_20171208_Archive_e001018.jpg" width="420"/>

**Seed (cleaned):**
> A massive phytoplankton bloom stained the waters of the Atlantic Ocean north of Iceland with brilliant jewel tones in late summer, 2014. The Moderate Resolution Imaging Spectroradiometer (MODIS) aboard NASA's Aqua satellite captured this true-color image on August 2. Huge colonies of the floating, plant-like organisms create swirls of green, teal and turquoise and cover over 80% of the visible ocean off the northeast coast of Iceland. Marine phytoplankton require just the right amount of sunlight, dissolved nutrients and water temperatures which are not too hot, nor too cold to spark explosive reproduction and result in blooms which can cover hundreds of square kilometers.

**Final caption** (687 chars):
> Imaged by the Moderate Resolution Imaging Spectroradiometer (MODIS) aboard the Aqua satellite, a massive phytoplankton bloom creates intricate, swirling patterns of green, teal, and turquoise across the Atlantic Ocean waters north of Iceland. The floating, plant-like organisms form a vast, marbled expanse that contrasts with the deep blue of the surrounding sea. White, spiraling clouds dominate the upper portion of the frame, while the snow-covered northern coast of Iceland is visible along the bottom edge. Marine phytoplankton reproduce explosively when sunlight, dissolved nutrients, and water temperatures align, resulting in blooms that can cover hundreds of square kilometers.

---

### GSFC_20171208_Archive_e001285  

<img src="images_thumb/GSFC_20171208_Archive_e001285.jpg" width="420"/>

**Seed (cleaned):**
> This shiny silver "waterfall" is actually the five layers of the full-scale engineering model of NASA's James Webb Space Telescope sunshield being laid out by technicians at the Northrop Grumman Aerospace Systems Space Park facility in Redondo Beach, Calif. who are conducting endurance tests on them.

**Final caption** (625 chars):
> The five layers of the full-scale engineering model of the James Webb Space Telescope sunshield are laid out in a large white hangar, with the material spread flat in the foreground and suspended vertically in the background. The thin film shimmers with silver and gold tones, resembling a massive metallic waterfall, while blue tape marks the edges of the layers near the floor. The vertical section is held up by cables from the ceiling, draping down in a tent-like shape. Technicians on platforms are visible near the base of the structure, conducting endurance tests on the sunshield, which was built by Northrop Grumman.

---

### GSFC_20171208_Archive_e001446  

<img src="images_thumb/GSFC_20171208_Archive_e001446.jpg" width="420"/>

**Seed (cleaned):**
> This image of tropical storm Andrea was assembled from data collected by NOAA's GOES-14 satellite at 8:31 a.m. EDT on June 7, when the storm's center was about 35 miles north-northwest of Charleston, S.C. Credit: NASA/NOAA GOES Project

**Final caption** (547 chars):
> Tropical Storm Andrea appears as a massive, swirling system of white clouds dominating the view over the Southeastern United States. Imaged by NOAA's GOES-14 satellite, the storm stretches from the Atlantic Ocean inland across the Carolinas and Virginia, displaying a broad, textured cloud cover. To the southwest, the dark blue waters of the Gulf of Mexico and the Florida peninsula are visible, along with the Caribbean islands of Cuba and Hispaniola in the lower right. A smaller, distinct cloud formation is visible near the Yucatan Peninsula.

---

### GSFC_20171208_Archive_e001565  

<img src="images_thumb/GSFC_20171208_Archive_e001565.jpg" width="420"/>

**Seed (cleaned):**
> Seen here we have started to encase the dinosaur footprints in what is known as a field jacket. A field jacket is much like a cast that a doctor would place on a broken arm or leg. Our field jacket consisted of many layers of burlap soaked in plaster-of-Paris into which we also laminate metal pipes to act like splints for additional support. Here Michael is working to remove the very hard sandstone layer below the iron-rich clay layer in which the prints were preserved.

**Final caption** (653 chars):
> A worker in a white protective suit and face shield kneels in an excavation pit, using a hand tool to carefully chip away at the side of a large, excavated earth block. The block is partially encased in a thick, brownish field jacket constructed from layers of burlap soaked in plaster-of-Paris and reinforced with metal pipes that act as splints for additional support. A blue roll of material sits atop the jacketed section, while sandbags and tarps line the upper edge of the pit. The worker is removing a hard sandstone layer located beneath the iron-rich clay layer where dinosaur footprints are preserved, preparing the fossil block for transport.

---

### KSC-314D-0502_0053  

<img src="images_thumb/KSC-314D-0502_0053.jpg" width="420"/>

**Seed (cleaned):**
> Operations are underway to remove two of the the Magnetospheric Multiscale spacecraft from their protective shipping container in the airlock of Building 2 at the Astrotech payload processing facility in Titusville, Florida, near Kennedy Space Center. The Magnetospheric Multiscale mission, or MMS, is a Solar Terrestrial Probes mission comprising four identically instrumented spacecraft that will use Earth's magnetosphere as a laboratory to study the microphysics of three fundamental plasma processes: magnetic reconnection, energetic particle acceleration and turbulence.

**Final caption** (766 chars):
> A large white transport container equipped with rows of green gas cylinders sits on the floor of a high-bay facility, where workers stand nearby. The wall behind the container displays the logo "ASTROTECH SPACE OPERATIONS," identifying the location as the Astrotech payload processing facility. In the foreground, a work table covered in clear plastic sheeting holds a large circular metal frame, purple lifting straps, and two white hard hats. This container houses the Magnetospheric Multiscale (MMS) spacecraft, part of a Solar Terrestrial Probes mission comprising four identically instrumented spacecraft. The mission uses Earth's magnetosphere as a laboratory to study the microphysics of magnetic reconnection, energetic particle acceleration, and turbulence.

---

### KSC-99pc0128  

<img src="images_thumb/KSC-99pc0128.jpg" width="420"/>

**Seed (cleaned):**
> Inside the launch tower at Pad 17-A, Cape Canaveral Air Station, workers guide the Stardust spacecraft toward an opening to a Boeing Delta II rocket below. The spacecraft is already connected to the third stage of the rocket that will be mated with the second stage in preparation for liftoff on Feb. 6. Stardust is destined for a close encounter with the comet Wild 2 in January 2004. Using a silicon-based substance called aerogel, Stardust will capture comet particles flying off the nucleus of the comet.

**Final caption** (526 chars):
> The Stardust spacecraft is suspended within a steel launch tower structure, wrapped in a large, shiny black protective covering. It hangs from a crane hook above a platform where several technicians in white cleanroom suits are working near equipment consoles. The spacecraft is being positioned for mating with a Boeing Delta II rocket, specifically connected to the third stage. Designed to capture particles from the tail of comet Wild 2 using a silicon-based aerogel, the mission involves a close encounter with the comet.

---

### LRC-1962-B701_P-05849  

<img src="images_thumb/LRC-1962-B701_P-05849.jpg" width="420"/>

**Seed (cleaned):**
> At blackboard, showing his space rendezvous concept for lunar landings. Lunar Orbital Rendezvous (LOR) would be used in the Apollo program.

**Final caption** (510 chars):
> A man in a white shirt and dark tie stands before a blackboard, pointing to a diagram labeled "L.O.R." under the heading "ESCAPE WEIGHTS." The chalk drawings include a "VEHICLE" schematic with "C.M." and "S.M." labels, and a comparison of "L.O.R." and "DIRECT" mission profiles. A trajectory map at the bottom traces a path from "EARTH" to "MOON" with numbered points. The presentation outlines the Lunar Orbital Rendezvous (LOR) concept, a strategy adopted for the Apollo program to facilitate lunar landings.

---

### LRC-2011-02552  

<img src="images_thumb/LRC-2011-02552.jpg" width="420"/>

**Seed (cleaned):**
> ORION Project-(SPLASH) Structural Passive Landing Attenuation for Survivability of Human Crew (BTA) Boiler Plate Test Article Water Impact Test-Pot Phase "0" Test Tested at the Hydro Impact Basin at the Landing and Impact Research Facility (Gantry)

**Final caption** (445 chars):
> The Orion Boiler Plate Test Article (BTA), a large white conical spacecraft component, is suspended by cables from a mobile crane. To the left, a tall, orange lattice gantry tower rises, with two workers in a lift basket positioned near the upper structure. The BTA is part of the SPLASH (Structural Passive Landing Attenuation for Survivability of Human Crew) project, designed to test the spacecraft's structural integrity during water impact.

---

### LRC-2011-03809  

<img src="images_thumb/LRC-2011-03809.jpg" width="420"/>

**Seed (cleaned):**
> UPWT Test 1998 Continuous Data Sonic Boom Test. Sonic Boom Hardward Mounted in the Langley Unitary Plan wind Tunnel(UPWT). Conical survey probes, wedge probe, and wind tunnel wall boundary layer rake. Rectangular box with wedge front end is a transducer box to that held pressure transducer for the conical probes.

**Final caption** (532 chars):
> A slender, metallic model with a pointed nose cone and tail fins extends horizontally from the right, its midsection wrapped in white material. Mounted on the large, circular metallic wall to the left is a white, wedge-shaped transducer box connected by white tubing, positioned above a horizontal probe and a vertical boundary layer rake. This sonic boom hardware is set up inside the Unitary Plan Wind Tunnel (UPWT) at Langley Research Center for continuous data collection. The yellow structural frame surrounds the test section.

---

### LRC-2012-02348  

<img src="images_thumb/LRC-2012-02348.jpg" width="420"/>

**Seed (cleaned):**
> SPLASH P1 Test 7 Swing Test: Documentation of preparation, set up and results of full scale BTA (Boilerplate Test Article) vertical drop test series performed in 2012 at the LaRC Hydro Impact Basin (HIB)

**Final caption** (559 chars):
> A white Boilerplate Test Article (BTA) hangs suspended from cables attached to a massive orange lattice crane at the Landing and Impact Research Facility. The facility is identified by a large white wall bearing the text "NASA LaRC LANDING AND IMPACT RESEARCH FACILITY." To the left, a large white arched hangar shelters equipment, including a yellow lift. In the foreground, the Hydro Impact Basin is a large pool of water bordered by yellow safety railings and white signs with black symbols. The setup is for the SPLASH P1 Test 7 vertical drop test series.

---

### LRC-2012-02449  

<img src="images_thumb/LRC-2012-02449.jpg" width="420"/>

**Seed (cleaned):**
> (NTF) National Transonic Facility Test 213-SFW Flow Control II, Fast-MAC Model: The fundamental Aerodynamics Subsonic Transonic-Modular Active Control (Fast-MAC) Model was tested for the 2nd time in the NTF. The objectives were to document the effects of Reynolds numbers on circulation control aerodynamics and to develop and open data set for CFD code validation. Image taken in building 1236, National Transonic Facility

**Final caption** (531 chars):
> A curved, brushed-metal leading edge component is attached to the smooth, gray fuselage of the Fast-MAC model, featuring a distinct semi-cylindrical profile and small gaps along its upper edge. The Fundamental Aerodynamics Subsonic Transonic-Modular Active Control (Fast-MAC) model is positioned inside the National Transonic Facility for aerodynamic testing. The experiment documents the effects of Reynolds numbers on circulation control aerodynamics and develops an open dataset for computational fluid dynamics code validation.

---

### NHQ202007300024  · scrubber applied

<img src="images_thumb/NHQ202007300024.jpg" width="420"/>

**Seed (cleaned):**
> In this black and white infrared image, a United Launch Alliance Atlas V rocket with NASA's Mars 2020 Perseverance rover onboard launches from Space Launch Complex 41, Thursday, July 30, 2020, at Cape Canaveral Air Force Station in Florida.

**Final caption** (518 chars):
> A United Launch Alliance Atlas V rocket carrying the Mars 2020 Perseverance rover ascends vertically into a dark sky, its engine glowing intensely at the apex and creating a bright halo effect. A thick, textured column of white exhaust trails downward from the vehicle, merging with a large, billowing cloud of smoke near the ground. Three slender vertical structures stand near the base of the plume against the dark background. The foreground consists of a textured, grassy terrain illuminated by the launch's light.

<details><summary>Pre-scrubber output</summary>

> A United Launch Alliance Atlas V rocket carrying NASA's Mars 2020 Perseverance rover ascends vertically into a dark sky, its engine glowing intensely at the apex and creating a bright halo effect. A thick, textured column of white exhaust trails downward from the vehicle, merging with a large, billowing cloud of smoke near the ground. Three slender vertical structures stand near the base of the plume against the dark background. The foreground consists of a textured, grassy terrain illuminated by the launch's light.

</details>

---

### PIA01996  

<img src="images_thumb/PIA01996.jpg" width="420"/>

**Seed (cleaned):**
> This image of Neptune south polar region was obtained by NASA Voyager on Aug. 23, 1989. The image shows the discovery of shadows in Neptune atmosphere, shadows cast onto a deep cloud bank by small elevated clouds.

**Final caption** (457 chars):
> Imaged by the Voyager spacecraft, the south polar region of Neptune displays a grainy, textured atmosphere with two bright, curved cloud bands running diagonally across the frame. Scattered bright specks, identified as small elevated clouds, cast distinct shadows onto the deeper cloud bank below. These shadows reveal the vertical structure of the atmosphere, indicating that high-altitude clouds are blocking sunlight from reaching the lower cloud layers.

---

### PIA03464  

<img src="images_thumb/PIA03464.jpg" width="420"/>

**Seed (cleaned):**
> The 2002 Winter Olympics are hosted by Salt Lake City at several venues within the city, in nearby cities, and within the adjacent Wasatch Mountains. This simulated natural color image presents a late spring view of north central Utah that includes all of the Olympic sites. The image extends from Ogden in the north, to Provo in the south; and includes the snow-capped Wasatch Mountains and the eastern part of the Great Salt Lake. This image was acquired on May 28, 2000 by the Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) on NASA's Terra satellite.

**Final caption** (549 chars):
> The dark blue waters of the Great Salt Lake occupy the left side of the frame, bordering the dense urban grid of Salt Lake City. To the east, the rugged, green slopes of the Wasatch Mountains rise steeply, dotted with patches of white snow on the higher peaks. The city sprawls through the valley floor, extending southward toward Utah Lake, visible as a bright blue body of water in the bottom right corner. This region of north central Utah hosted the 2002 Winter Olympics, with venues located throughout the city and the adjacent mountain ranges.

---

### PIA03502  

<img src="images_thumb/PIA03502.jpg" width="420"/>

**Seed (cleaned):**
> Over 1300 energy spectra taken on September 22, 2001 from the ion and electron instruments on NASA Deep Space 1 span a region of 1,400,000 kilometers 870,000 miles centered on the closest approach to the nucleus of comet Borrelly.

**Final caption** (511 chars):
> A plot of energy spectra features horizontal bands of red, yellow, and green stretching across a black background, intersected by a large, V-shaped structure of green and yellow hues extending vertically from the center. Vertical streaks of blue and purple texture the upper and lower sections. The data represents measurements taken by the ion and electron instruments on the Deep Space 1 spacecraft. These spectra span the region surrounding the spacecraft's closest approach to the nucleus of comet Borrelly.

---

### PIA12657  

<img src="images_thumb/PIA12657.jpg" width="420"/>

**Seed (cleaned):**
> Light and dark terrain covers the surface of Saturn moon Iapetus in this view from NASA Cassini spacecraft.

**Final caption** (291 chars):
> Saturn's moon Iapetus, imaged by the Cassini spacecraft, appears as a crescent against the black background of space. Its surface is a rugged landscape of impact craters, displaying a mix of light and dark terrain. The illuminated right side is bright, while the left side fades into shadow.

---

### PIA13039  

<img src="images_thumb/PIA13039.jpg" width="420"/>

**Seed (cleaned):**
> A sinuous rille created by a lava flow snakes around the base of a massif in the Prinz-Harbinger region on the Moon in this image taken by NASA Lunar Reconnaissance Orbiter.

**Final caption** (393 chars):
> Imaged by the Lunar Reconnaissance Orbiter, a dark, sinuous rille winds through a heavily cratered lunar landscape, its deep, shadowed channel contrasting sharply with the lighter gray regolith. The rille curves from the upper center down toward the bottom right, snaking around the base of a large, elevated massif. This feature in the Prinz-Harbinger region was carved by ancient lava flows.

---

### PIA13330  

<img src="images_thumb/PIA13330.jpg" width="420"/>

**Seed (cleaned):**
> This mosaic, made from radar images obtained by NASA Cassini spacecraft, shows mountainous terrain on Saturn moon Titan in the moon northern hemisphere, north of the Aaru region.

**Final caption** (465 chars):
> Rough, mountainous terrain on Saturn's moon Titan is mapped in a grayscale radar mosaic, characterized by a mottled texture of light and dark gray patches. Two curved bands of terrain are separated by white space, overlaid with thin, vertical lines of color ranging from blue to orange. These colored lines represent the individual radar swaths collected by the Cassini spacecraft. The terrain is located in the moon's northern hemisphere, north of the Aaru region.

---

### PIA16609  

<img src="images_thumb/PIA16609.jpg" width="420"/>

**Seed (cleaned):**
> This artist conception illustrates the brown dwarf named 2MASSJ22282889-431026. NASA Hubble and Spitzer space telescopes observed the object to learn more about its turbulent atmosphere.

**Final caption** (409 chars):
> The brown dwarf 2MASSJ22282889-431026 is a reddish-purple sphere with horizontal bands of swirling orange and yellow clouds. The atmosphere features turbulent striations and varying shades of red and pink across its surface. This artist's conception visualizes the object based on data from the Hubble and Spitzer space telescopes, which observed the brown dwarf to study its turbulent atmospheric conditions.

---

### PIA17172  · scrubber applied

<img src="images_thumb/PIA17172.jpg" width="420"/>

**Seed (cleaned):**
> On July 19, 2013, in an event celebrated the world over, NASA's Cassini spacecraft slipped into Saturn's shadow and turned to image the planet, seven of its moons, its inner rings -- and, in the background, our home planet, Earth. The outermost ring shown here is Saturn's E ring, the core of which is situated about 149,000 miles (240,000 kilometers) from Saturn. The geysers erupting from the south polar terrain of the moon Enceladus supply the fine icy particles that comprise the E ring; diffraction by sunlight gives the ring its blue color.

**Final caption** (535 chars):
> Saturn is silhouetted as a dark sphere at the center, surrounded by a system of golden-brown rings with distinct bands and gaps. A faint, diffuse blue halo extends outward from the ring system, identified as the E ring. Imaged by the Cassini spacecraft as it slipped into Saturn's shadow, the planet in darkness while the rings are backlit. The blue color of the E ring results from diffraction by sunlight, and the ring is composed of fine icy particles supplied by geysers erupting from the south polar terrain of the moon Enceladus.

<details><summary>Pre-scrubber output</summary>

> Saturn is silhouetted as a dark sphere at the center, surrounded by a system of golden-brown rings with distinct bands and gaps. A faint, diffuse blue halo extends outward from the ring system, identified as the E ring. Imaged by the Cassini spacecraft as it slipped into Saturn's shadow, the view captures the planet in darkness while the rings are backlit. The blue color of the E ring results from diffraction by sunlight, and the ring is composed of fine icy particles supplied by geysers erupting from the south polar terrain of the moon Enceladus.

</details>

---

### PIA23889  

<img src="images_thumb/PIA23889.jpg" width="420"/>

**Seed (cleaned):**
> This animated GIF shows the deployment of the Perseverance rover's remote sensing mast during a cold test in a space simulation chamber at NASA's Jet Propulsion Laboratory. The test took place in October 2019.

**Final caption** (472 chars):
> The Perseverance rover's remote sensing mast sits on a large, black, ribbed circular platform inside a space simulation chamber. The complex assembly features white instrument housings, metallic structural elements, and a dense network of red, black, and yellow wiring connecting various sensors and mechanisms. A vertical support structure braces the mast as it undergoes a cold test at Jet Propulsion Laboratory to verify its functionality in simulated space conditions.

---

### S111E5034  

<img src="images_thumb/S111E5034.jpg" width="420"/>

**Seed (cleaned):**
> Astronaut Franklin R. Chang-Diaz works with a grapple fixture during extravehicular activity (EVA) to perform work on the International Space Station (ISS). The first spacewalk of the STS-111 mission began with the installation of a Power and Data Grapple Fixture (PDGF) for the station's robotic arm on the complex's P6 truss. The PDGF will allow the robotic arm to grip the P6 truss for future station assembly operations.

**Final caption** (561 chars):
> An astronaut in a white spacesuit floats near the International Space Station, working on a circular metallic fixture attached to a truss structure covered in white thermal blankets. The astronaut is installing a Power and Data Grapple Fixture (PDGF) during the STS-111 mission, a component that allows the station's robotic arm to grip the P6 truss for future assembly operations. The background reveals the bright blue curve of Earth's atmosphere against the blackness of space, while cables and connectors hang from the station's framework in the foreground.

---

### STS059-S-001  

<img src="images_thumb/STS059-S-001.jpg" width="420"/>

**Seed (cleaned):**
> Designed by the crew members, the STS-59 insignia is dominated by Earth, reflecting the focus of the first Space Radar Laboratory (SRL-1) mission upon our planet's surface and atmosphere. The golden symbol of the astronaut corps emblem sweeps over Earth's surface from the space shuttle Endeavour, representing the operation of the SIR-C/Synthetic Aperture Radar (X-SAR) and the Measurement of Air Pollution from Space (MAPS) sensors. The astronaut emblem also signals the importance of the human element in space exploration and in the study of our planet.

**Final caption** (737 chars):
> The STS-59 mission patch features a central illustration of the Space Shuttle Endeavour in silhouette against a starry black background, with a blue globe showing the continents of North and South America behind it. A golden astronaut corps emblem, consisting of a star atop three vertical bars encircled by a ring, rises from the shuttle's payload bay. The red outer ring contains the names of the crew members in white capital letters: CLIFFORD, APT, GUTIERREZ, CHILTON, NINOW, GODWIN, and JONES. The mission designation "SRL-1" is printed in black at the bottom. Designed by the crew, the insignia reflects the focus of the first Space Radar Laboratory mission on Earth's surface and atmosphere using the SIR-C/X-SAR and MAPS sensors.

---

### as13-59-8501  

<img src="images_thumb/as13-59-8501.jpg" width="420"/>

**Seed (cleaned):**
> This view of the severely damaged Apollo 13 Service Module (SM) was photographed from the Lunar Module/Command Module (LM/CM) following SM jettisoning. As seen here, an entire panel on the SM was blown away by the apparent explosion of oxygen tank number two located in Sector 4 of the SM. Two of the three fuel cells are visible just forward (above) the heavily damaged area. Three fuel cells, two oxygen tanks, and two hydrogen tanks are located in Sector 4. The damaged area is located above the S-Band high gain antenna.

**Final caption** (485 chars):
> The Apollo 13 Service Module floats in the blackness of space, its cylindrical body marred by a large, jagged hole where an entire panel was blown away by the explosion of oxygen tank number two. The damaged area, located in Sector 4 above the S-Band high-gain antenna, exposes the internal structure of the spacecraft. Two of the three fuel cells remain visible just forward of the heavily damaged section, while the rest of the module's exterior is illuminated against the dark void.

---

### as14-66-09305  

<img src="images_thumb/as14-66-09305.jpg" width="420"/>

**Seed (cleaned):**
> A front view of the Apollo 14 Lunar Module (LM), which reflects a circular flare caused by the brilliant sun, as seen by the two moon-exploring crew men (out of frame) of the Apollo 14 lunar landing mission during their first extravehicular activity (EVA). The unusual ball of light was said by the astronauts to have a jewel-like appearance. In the left background Cone Crater can be seen. In the left foreground are the erectable S-Band antenna and the United States flag.

**Final caption** (698 chars):
> The Apollo 14 Lunar Module sits on the gray lunar surface, its gold-foil-wrapped descent stage and landing legs illuminated by a brilliant sun positioned directly behind it. A massive, circular lens flare dominates the upper center of the frame, creating a halo effect around the spacecraft's upper assembly. To the left, a white, umbrella-shaped S-Band antenna stands next to a United States flag planted in the regolith. The text "UNITED STATES" is visible on a side panel of the module. In the distance, the rim of Cone Crater is visible against the dark sky. The Lunar Module served as the landing vehicle for the Apollo 14 mission, photographed during the crew's first extravehicular activity.

---

### cloud-free-view-of-iceland_17168825861_o  

<img src="images_thumb/cloud-free-view-of-iceland_17168825861_o.jpg" width="420"/>

**Seed (cleaned):**
> This nearly cloud-free image of Iceland was captured by the MODIS instrument on board the Terra spacecraft on 04/15/2015 at 13:00 UTC.

**Final caption** (526 chars):
> The island of Iceland is blanketed in snow and ice, its white surface standing out against the deep blue ocean. The western coastline is characterized by deep, branching fjords, while the rest of the perimeter is rugged and irregular. Wispy clouds streak across the northern portion of the island and the surrounding waters, with additional scattered clouds in the lower left. A dark, rectangular body of water is visible within the ice sheet in the southeastern region. Imaged by the MODIS instrument on the Terra spacecraft.

---

### iss040e072900  

<img src="images_thumb/iss040e072900.jpg" width="420"/>

**Seed (cleaned):**
> Image taken during BASS-II flame test session -card 2 - conducted on GMT 203. Burned a 4 mm thick, 1 cm wide acrylic slab at multiple velocities. At high flows, the flame was long and sooty. When the flow was reduced, the flame stabilized. The Burning and Suppression of Solids II (BASS-II) investigation examines the burning and extinction characteristics of a wide variety of fuel samples in microgravity.

**Final caption** (572 chars):
> A glowing, oval-shaped flame burns against a black background, featuring a bright blue outer rim and a diffuse pinkish-orange center. A dark, elongated shape protrudes from the right side into the flame, marking the position of a 4 mm thick, 1 cm wide acrylic slab. This test is part of the Burning and Suppression of Solids II (BASS-II) investigation, which examines the burning and extinction characteristics of various fuel samples in microgravity. The flame's stabilized appearance, with its distinct blue boundary, indicates a reduced flow rate during the experiment.

---

## DROP branch — caption from pixels only (seed discarded)  _(32)_

### 0201797  

<img src="images_thumb/0201797.jpg" width="420"/>

**Seed (cleaned):**
> Astronaut James H. Newman, mission specialist, floats about in the Space Shuttle Columbia's cargo bay while working in tandem with astronaut Michael J. Massimino (out of frame), mission specialist, during the STS-109 mission's second day of extravehicular activity (EVA). Inside Columbia's cabin, astronaut Nancy J. Currie, mission specialist, controlled the Remote Manipulator System (RMS) to assist the two in their work on the Hubble Space Telescope (HST). The RMS was used to capture the telescope and secure it into Columbia's cargo bay.

**Final caption** (657 chars):
> An astronaut in a white spacesuit floats in the foreground, reaching toward a piece of equipment wrapped in gold foil insulation. The astronaut wears a helmet with a gold visor reflecting Earth, an American flag patch on the left shoulder, and a NASA logo on the chest. The equipment box features a printed American flag and a NASA logo on its front face, secured with silver latches. Behind the astronaut, a large white spacecraft structure is visible with black panels and yellow handrails marked with the numbers "240," "270," and "300." A mechanism labeled "YAW" is visible on the right side, and the blackness of space appears in the upper left corner.

---

### 0701576  

<img src="images_thumb/0701576.jpg" width="420"/>

**Seed (cleaned):**
> Under the goals of the Vision for Space Exploration, Ares I is a chief component of the cost-effective space transportation infrastructure being developed by NASA's Constellation Program. This transportation system will safely and reliably carry human explorers back to the moon, and then onward to Mars and other destinations in the solar system. The Ares I effort includes multiple project element teams at NASA centers and contract organizations around the nation, and is managed by the Exploration Launch Projects Office at NASA's Marshall Space Flight Center (MFSC).

**Final caption** (496 chars):
> A large, flat panel covered in a green material with a dense pattern of small white dots lies on a work surface. The panel is divided into large rectangular sections by thin, darker lines and is framed by a light-colored wooden border. A person wearing a blue shirt uses a red brush to smooth or treat the surface near the top right corner, while their other hand rests on the edge of the frame. The background reveals a tiled floor with dark stripes, suggesting a workshop or laboratory setting.

---

### 200907200085HQ  

<img src="images_thumb/200907200085HQ.jpg" width="420"/>

**Seed (cleaned):**
> Veteran journalist Nick Clooney, center, moderated a panel discussion with Apollo 11 astronaut Buzz Aldrin, far right, Charlie Duke of Apollo 16, John Grunsfeld, of the recent Hubble mission, not seen, and Goddard Space Flight Center deputy director Laurie Leshin, Monday, July 20, 2009, at the Newseum in Washington as part of the commemoration of the 40th Anniversary of the Apollo 11 moon landing.

**Final caption** (539 chars):
> A man with white hair, wearing a dark suit and a patterned tie, sits facing a group of three people seated in the foreground with their backs to the camera. To the left, a podium displays a blue sign with the partial white text "NEWSE". The central figure has his hands clasped in his lap and is looking toward the man on the left. The listeners include two men with white hair in dark jackets and a woman with reddish-brown hair in a light blue top on the far right. The background is completely black, isolating the group in a spotlight.

---

### 201202200041HQ  

<img src="images_thumb/201202200041HQ.jpg" width="420"/>

**Seed (cleaned):**
> Captain Mark Kelly, commander of the space shuttle Endeavour's final mission and husband of retired U.S. Representative Gabrielle Giffords, gives the keynote address during a celebration dinner at Ohio State University honoring the 50th anniversary of John Glenn's historic flight aboard Friendship 7 Monday, Feb. 20, 2012, in Columbus, Ohio.

**Final caption** (588 chars):
> A bald man in a dark suit and light blue tie stands on a raised stage, gesturing with both hands while speaking. Behind him, large screens display a live feed of his speech on the left and a black-and-white historical photograph of a crowd beneath a white airship on the right. An American flag and a large arrangement of white lilies and blue flowers stand near the center of the stage, partially obscuring a camera operator filming the event. The stage is bathed in deep blue light, with black skirting along the base and polished wooden flooring in the foreground reflecting the scene.

---

### 201203020010HQ  

<img src="images_thumb/201203020010HQ.jpg" width="420"/>

**Seed (cleaned):**
> Wife of former astronaut and Senator John Glenn, Annie Glenn, listens intently to Cleveland State University Master of Music Major James Binion Jr. as he sings a musical tribute during an event celebrating John Glenn's legacy and 50 years of americans in orbit held at the university's Wolstein Center on Friday, March 3, 2012 in Cleveland, Ohio.

**Final caption** (379 chars):
> An elderly woman with short, curly white hair looks upward and to the left, her right hand raised near her chin with fingers slightly curled. She wears a bright red coat with a gold brooch pinned to the lapel and a gold ring on her finger. Her face is illuminated against a dark background, where the shoulder of another person in a dark garment is partially visible on the left.

---

### 201308060009HQ  

<img src="images_thumb/201308060009HQ.jpg" width="420"/>

**Seed (cleaned):**
> Expedition 35/36 NASA astronaut Chris Cassidy, left on screen, is seen on a live feed from the International Space Station as they participate in a public event at NASA Headquarters observing the first anniversary of the Curiosity rover's landing on Mars, Tuesday, August 6th, 2013 in Washington.

**Final caption** (748 chars):
> Two models of Mars rovers are displayed on a stage in front of a blue banner featuring a repeating list of NASA center names, including STENNIS, AMES, DRYDEN, GLENN, GODDARD, HEADQUARTERS, MARSHALL, JPL, JOHNSON, and LANGLEY. On the left, a smaller rover with a gold-foil body, a white instrument mast, and a dish antenna sits near a spherical model of Mars on a black pedestal. To the right, a larger white rover with large, treaded black wheels and a complex mast stands on a simulated rocky surface. Above the banner, a large screen shows a video feed of two people inside a spacecraft module, flanked by two large NASA logos. In the foreground, audience members watch the presentation, with one person holding up a smartphone to take a picture.

---

### 201409170017HQ  

<img src="images_thumb/201409170017HQ.jpg" width="420"/>

**Seed (cleaned):**
> Panelist, from left, Lisa May, lead program executive, Mars Exploration Program, NASA Headquarters, Washington, Bruce Jakosky, MAVEN principal investigator, Laboratory for Atmospheric and Space Physics, University of Colorado, Boulder, David Mitchell, MAVEN project manager, NASA's Goddard Space Flight Center, Greenbelt, Maryland, and Guy Beutelschies, Lockheed Martin MAVEN program manager, Lockheed Martin Space Systems Company, Littleton, Colorado, all shake hands at the end of a media briefing.

**Final caption** (665 chars):
> Four people sit behind a long, glossy black table, engaging in a continuous chain of handshakes across the surface. From left to right, a woman with short red hair in a brown jacket shakes hands with a bearded man in a dark blue shirt, who in turn shakes hands with a man in a tan blazer, who shakes hands with a man in a black polo shirt on the far right. Each person has a gooseneck microphone and a mug in front of them, and a large blue and red NASA logo is mounted on the front panel of the table. The backdrop features a space scene showing a spacecraft with large solar panels orbiting a reddish, cratered planet, with Earth visible in the upper left corner.

---

### 201503190009HQ  

<img src="images_thumb/201503190009HQ.jpg" width="420"/>

**Seed (cleaned):**
> Expedition 43 Russian Cosmonaut Gennady Padalka of the Russian Federal Space Agency (Roscosmos) talks with an instructor during an Soyuz TMA-16M Trajectory and Flight Plan briefing, Thursday, March 19, 2015 at Cosmonaut Hotel in Baikonur, Kazakhstan.

**Final caption** (397 chars):
> A man with light brown hair and wire-rimmed glasses leans forward with his arms resting on a table, looking intently toward a person in the foreground who is out of focus. He wears a blue polo shirt featuring a logo on the left chest with the letters "AZ" and a stylized graphic. A watch is visible on his left wrist. Black office chairs are visible in the background against a light-colored wall.

---

### 9407055  

<img src="images_thumb/9407055.jpg" width="420"/>

**Seed (cleaned):**
> The second manned lunar landing mission, Apollo 12, launched from launch pad 39-A at Kennedy Space Center. The Saturn V vehicle was developed by the Marshall Space Flight Center under the direction of Dr. Wernher von Braun. Aboard Apollo 12 was a crew of three astronauts: Alan L. Bean, Richard Gordon, and Spacecraft Commander Charles Conrad. Lunar soil activities included the deployment of the Apollo Lunar Surface Experiments Package (ALSEP), finding the unmanned Surveyor 3 that landed on the Moon on April 19, 1967, and collecting 75 pounds (34 kilograms) of rock samples. Apollo 12 safely returned to Earth on November 24, 1969.

**Final caption** (518 chars):
> The gold-foil-wrapped landing legs and lower section of a spacecraft rest on a gray, dusty surface. The legs feature tubular struts covered in reflective gold material, terminating in flat, circular footpads. The main body is wrapped in gold and white insulation blankets, with mechanical linkages and cables visible near the base, alongside a white cylindrical object. In the foreground, distinct footprints and tracks mark the uneven terrain, while a curved horizon line separates the gray ground from the black sky.

---

### AFRC2018-0287-278  

<img src="images_thumb/AFRC2018-0287-278.jpg" width="420"/>

**Seed (cleaned):**
> NASA mission controllers, engineers, pilots and communications specialists in the mission control room monitor the supersonic research flight off the coast of Galveston, as part of the QSF18 flight series. The flight operations crew tracks the status of the flights, maintains communications with the aircraft, communicates with U.S. Coast Guard, and coordinates community feedback data.

**Final caption** (708 chars):
> A man in a light blue short-sleeved shirt sits in a black office chair at a white folding table, facing away from the camera while monitoring multiple computer screens. In the foreground, an open laptop displays a software interface featuring a map with a blue polygon and data fields. Behind the laptop, a large monitor shows a satellite map with a large orange polygon overlaid on a coastal area, while a smaller screen to the right displays a similar map view. To the left, a dark monitor reflects the room's interior. The setting appears to be a temporary facility or control center, characterized by white fabric walls, exposed metal conduits, and shelves holding electronic equipment in the background.

---

### ARC-2009-ACD09-0220-057  

<img src="images_thumb/ARC-2009-ACD09-0220-057.jpg" width="420"/>

**Seed (cleaned):**
> LCROSS Impact Night: From left to Right Jack Boyd, unknown, Pete Klupar, (middle) Lewis Braxton III, Pete Worden, John Marmie (back) Tina Panontin (seated), Chuck Duff and unknown in the Mission Operations Control Room (MOCR).

**Final caption** (598 chars):
> A group of nine people poses for a photograph in an office setting, arranged in rows around computer workstations. In the foreground, three men sit in purple office chairs; the central figure wears a black jacket and blue jeans, flanked by an older man in a black jacket on the left and a man in a beige shirt on the right. Behind them, six other individuals stand or sit near desks equipped with computer monitors displaying data. A large flat-screen television mounted on the wall shows a video feed of a room with desks and equipment. Several people wear lanyards and pink identification badges.

---

### ARC-2010-ACD10-0015-109  

<img src="images_thumb/ARC-2010-ACD10-0015-109.jpg" width="420"/>

**Seed (cleaned):**
> Ames 70th Anniversary Turn Back the Clock Day with antique Class Cars and Classic Car parade around the center. Lewis Braxton III, Ames Deputy Center Director and Jack Boyd, Ames Historian lead the parade in Lew's Mustang.

**Final caption** (480 chars):
> An orange Ford Mustang with black racing stripes on the hood drives forward on a paved road, its chrome bumper and round headlights prominent. The car features a large hood scoop and a horizontal grille. Inside, two men wearing hats are visible; the passenger on the right raises a hand in a wave. A California license plate reading "LSB 70" with a "MUSTANG" frame is mounted on the front bumper. Behind the orange car, a black convertible and several white SUVs follow in a line.

---

### EC00-0226-21  

<img src="images_thumb/EC00-0226-21.jpg" width="420"/>

**Seed (cleaned):**
> Following initial captive flight tests last year at NASA's Dryden Flight Research Center, Edwards Air Force Base, California, the X-34 technology demonstrator began a new series of tests last week in which it is being towed behind a semi-truck and released to coast on the Edwards dry lakebed. On July 20, 2000, it was towed and released twice at speeds of five and 10 miles per hour. On July 24, 2000, it was towed and released twice at 10 and 30 miles per hour.

**Final caption** (474 chars):
> A large white Peterbilt semi-truck tractor unit stands in the foreground on a vast, flat concrete surface, its chrome grille and vertical exhaust stacks gleaming in the sunlight. Further back on the tarmac, a small gray single-engine aircraft is parked with its landing gear down, flanked by white support vehicles and a small group of people standing nearby. The expansive, light-colored ground stretches toward a horizon line of low, hazy mountains under a clear blue sky.

---

### EC02-0273-5  

<img src="images_thumb/EC02-0273-5.jpg" width="420"/>

**Seed (cleaned):**
> Navajo Code Talker Joe Morris, Sr. shared insights from his time as a secret World War Two messenger with his audience at NASA's Dryden Flight Research Center on Nov. 26, 2002. NASA Dryden is located on Edwards Air Force Base in California's Mojave Desert.

**Final caption** (643 chars):
> An older man with gray hair and glasses stands behind a podium, wearing a tan uniform shirt and a red garrison cap. The cap features the gold Eagle, Globe, and Anchor emblem and the yellow embroidered text "NAVAJO CODE TALKERS." He wears a bolo tie with a large, circular pendant displaying a red and yellow sunburst design surrounding a central Marine Corps emblem. His uniform is adorned with collar insignia, a row of service ribbons, and medals on the left chest, along with a patch on the left sleeve. An American flag hangs vertically to his left against a blue curtain backdrop, and two black microphones are positioned in front of him.

---

### EC04-0269-1  

<img src="images_thumb/EC04-0269-1.jpg" width="420"/>

**Seed (cleaned):**
> 2004 NASA Dryden Research Pilots. Left to Right: Edwin W. Lewis, Jr., David A. Wright (Director of Flight Operations), William Frederick Brockett, Frank Batteas, Craig R. Bomben, Richard G. Ewers, James W. Smolka, Douglas H. Baker, C. Gordon Fullerton (Chief Pilot), James Barrilleaux, Martin J. Trout, and Mark Pestana. (not pictured: Dana Purifoy)

**Final caption** (704 chars):
> A group of eleven men in tan flight suits stands in a row on a concrete tarmac, posing in front of two parked aircraft. Behind them, a large white hangar features a blue NASA logo centered above a series of tall, grid-patterned windows, with the number 4802 visible on the upper left corner. To the left sits a white twin-engine turboprop plane with a blue stripe along the fuselage, displaying "NASA" and the number 7 on its tail fin. To the right is a sleek, white and black experimental aircraft with a pointed nose and a large external fuel tank mounted beneath the fuselage, bearing "NASA" and the number 850 on its vertical stabilizer. The scene is set under a clear blue sky with scattered clouds.

---

### ECN-1203  

<img src="images_thumb/ECN-1203.jpg" width="420"/>

**Seed (cleaned):**
> Fred W. Haise Jr. was a research pilot and an astronaut for the National Aeronautics and Space Administration from 1959 to 1979. He began flying at the Lewis Research Center in Cleveland, Ohio (today the Glenn Research Center), in 1959. He became a research pilot at the NASA Flight Research Center (FRC), Edwards, Calif., in 1963, serving NASA in that position for three years until being selected to be an astronaut in 1966. After becoming an astronaut, Haise served as a backup crewmember for the Apollo 8, 11, and 16 missions. He flew on the aborted Apollo 13 lunar mission in 1970, spending 142 hours and 54 minutes in space before returning safely to Earth.

**Final caption** (611 chars):
> A man with short dark hair stands smiling in front of a large blue aircraft fuselage, wearing a gray flight suit with an orange shirt visible at the collar. A black name tag on his left chest reads "FRED W. HAISE" above "TEST PILOT" and "NASA," while a mission patch is visible on his right chest and pens are tucked into his left breast pocket. Behind him, a large NASA logo with a red chevron and white lettering is painted on the blue surface, next to a large, dark jet engine intake rimmed with rivets. To the left, the arm and shoulder of another person in a similar gray flight suit are partially visible.

---

### GRC-1959-C-50990  

<img src="images_thumb/GRC-1959-C-50990.jpg" width="420"/>

**Seed (cleaned):**
> NASA photographer Arthur Laufman sets up a camera to film a Mercury capsule. The Photographic Branch, referred to as the Photo Lab, was part of the center's Technical Reports Division. Laufman joined the Photo Lab staff in 1948 and began producing full-length technical films as a tool to educate those outside of the agency on the research being conducted at Lewis. He worked with engineers to determine proper subjects for these films and develop a script.

**Final caption** (625 chars):
> A large, silver, ribbed cylinder, resembling a spacecraft component, is mounted on a dark, wheeled transport cart and tilted upward. The structure features horizontal corrugations, rows of rivets along the seams, and a series of circular cutouts around the rear rim, while the front opening reveals a dark interior with internal framework. To the left, a man in a light-colored shirt and dark trousers leans over a camera mounted on a tripod, aiming it at the object. A black equipment case sits on the paved ground near the cart's support leg, and the rear of a vintage car is visible in the background near a line of trees.

---

### GRC-2019-C-05316  

<img src="images_thumb/GRC-2019-C-05316.jpg" width="420"/>

**Seed (cleaned):**
> The inaugural Glenn Symposium focused on advancements in aerospace technology including power and propulsion, autonomy and communications, low boom supersonics, hypersonics, and more. Discussion also encompassed humans returning to the moon, including challenges associated with the 2024 mission.

**Final caption** (625 chars):
> A group of people stands on a light gray floor inside a large industrial facility, observing a massive metallic structure. The structure features a skeletal, cage-like framework supporting a large, dark, circular component at the top, with a blue NASA logo visible on the side. An orange aerial work platform is positioned near the base of the structure, its boom arm extended slightly. To the right, a man in a blue plaid shirt gestures toward the assembly while speaking to the group. The facility has tall concrete walls, with metal scaffolding attached to the left wall and large black curtains hanging in the background.

---

### GSFC_20171208_Archive_e000302  

<img src="images_thumb/GSFC_20171208_Archive_e000302.jpg" width="420"/>

**Seed (cleaned):**
> Versatile, tough and ready for mission-specific modification, NASA's C130 can carry scientists, equipment, cargo or combinations of all three. The North Atlantic Aerosols and Marine Ecosystems Study (NAAMES) is a five year investigation to resolve key processes controlling ocean system function, their influences on atmospheric aerosols and clouds and their implications for climate. Michael Starobin joined the NAAMES field campaign on behalf of Earth Expeditions and NASA Goddard Space Flight Center's Office of Communications. He presented stories about the important, multi-disciplinary research being conducted by the NAAMES team, with an eye towards future missions on the NASA drawing board. This is a NAAMES photo essay put together by Starobin, a collection of 49 photographs and captions.

**Final caption** (511 chars):
> A gray C-130J Super Hercules transport aircraft sits on a concrete tarmac, its four turboprop engines equipped with large, six-bladed propellers. The aircraft's side door is open, revealing rows of seats inside the fuselage, while a small white utility vehicle is parked nearby on the ground. To the left, a white ground support trailer labeled "TLD" and "TLD Green" is positioned near the nose gear. In the background, a teal-colored building and the tail of another aircraft are visible under an overcast sky.

---

### GSFC_20171208_Archive_e000559  

<img src="images_thumb/GSFC_20171208_Archive_e000559.jpg" width="420"/>

**Seed (cleaned):**
> As part of her visit to the United States, President Park Geun-hye of South Korea visited NASA's Goddard Space Flight Center in Greenbelt, Md. On Oct. 14, 2015. The visit offered an opportunity to celebrate past collaborative efforts between the American and South Korean space programs along with presentations on current projects and programs underway at Goddard.

**Final caption** (645 chars):
> A man in a dark suit holding a microphone stands behind a long table covered in black cloth, addressing a group of people in a large facility with high ceilings and blue safety fencing. On the table, several computer monitors, blue equipment cases, and a small white stool holding a small robotic device with visible circuitry and wheels are arranged. A brochure on the table reads "CANYVAL-2 CubeSat Mission," and a sign in the background displays "CANYVAS." Several people in business attire, including a woman in a yellow patterned dress and a man in the foreground right wearing glasses, stand around the table listening to the presentation.

---

### KSC-07PD-0704  

<img src="images_thumb/KSC-07PD-0704.jpg" width="420"/>

**Seed (cleaned):**
> In a clean-room environment at North Vandenberg Air Force Base, lights are reflected on the solar array panels of the AIM spacecraft during illumination testing. The AIM spacecraft will fly three instruments designed to study those clouds located at the edge of space. The mission's primary goal is to explain why these clouds form. AIM is scheduled to be mated to the Pegasus XL during the second week of April, after which final inspections will be conducted. Launch is scheduled for April 25.

**Final caption** (662 chars):
> A large, boxy spacecraft with deployed solar arrays rests on a circular metallic turntable inside a clean room facility. The central body and two side wings are densely packed with black rectangular solar cells, outlined by strips of yellow tape along the edges. Clear plastic sheeting hangs from the overhead frame, creating a barrier around the assembly area. To the right, a technician in a white full-body cleanroom suit stands near the satellite, facing slightly toward the camera. Black cables and hoses connect to the base of the spacecraft, draping down onto the platform, while small tools and yellow markers lie on the floor in the bottom right corner.

---

### KSC-2013-3337  

<img src="images_thumb/KSC-2013-3337.jpg" width="420"/>

**Seed (cleaned):**
> At the Naval Station Norfolk near NASA's Langley Research Center in Virginia, the Orion boilerplate test article has been returned to a U.S. Navy ship following a stationary recovery test in the water. The stationary recovery test will allow the teams to demonstrate and evaluate the recovery processes. The first unpiloted test flight of the Orion is scheduled to launch in 2014 atop a Delta IV rocket and in 2017 on NASA's Space Launch System rocket.

**Final caption** (600 chars):
> A large gray naval vessel dominates the frame with its stern hangar bay door lowered, revealing a dark interior illuminated by overhead lights. Inside the bay, crew members in orange life vests stand near large yellow and red spherical fenders, while a red flag hangs from the right side of the opening. In the foreground, a small rigid-hulled inflatable boat labeled "UNITED STATES" carries several crew members wearing orange life vests and camouflage trousers, navigating the choppy green water near the ship's hull. To the right, a smaller black inflatable boat with crew members is also visible.

---

### KSC-2014-4685  

<img src="images_thumb/KSC-2014-4685.jpg" width="420"/>

**Seed (cleaned):**
> SAN DIEGO, Calif. – On the third day of preparations for recovery of Orion after its splashdown in the Pacific Ocean, the well deck of the USS Anchorage has been filled with water and recovery hardware is in place. U.S. Navy divers have embarked from ship to practice recovery procedures in rigid hull inflatable boats and Zodiac boats. NASA, Lockheed Martin and U.S. Navy personnel are preparing for recovery of the crew module, forward bay cover and parachutes on its return from space and splashdown in the Pacific Ocean. The first unpiloted flight test of Orion is scheduled to launch Dec. 4 atop a United Launch Alliance Delta IV Heavy rocket.

**Final caption** (607 chars):
> The interior of a large ship's bay is filled with dark, choppy water that extends toward the open ocean visible through the stern. A yellow safety net is suspended across the opening, hovering just above the water line. On the upper right deck, a group of people wearing orange life vests stands along the railing, observing the water and the sea beyond. The ship's metallic walls are riveted and lined with pipes and equipment, while overhead lights illuminate the space. Through the open rear, small boats are visible on the horizon. A dark walkway with red bollards runs along the left side of the frame.

---

### KSC-20200728-PH-JBS02_0010  

<img src="images_thumb/KSC-20200728-PH-JBS02_0010.jpg" width="420"/>

**Seed (cleaned):**
> From NASA Kennedy Space Center's Press Site lawn near the iconic countdown clock, storm clouds can be seen rolling in over the Vehicle Assembly Building (VAB) on July 28, 2020. Standing at 525 feet tall, this facility is capable of hosting multiple varieties of rockets and spacecraft at the same time. Currently, the VAB is being utilized to process and assemble the agency's Space Launch System (SLS) rocket for Artemis missions to the Moon and beyond to Mars. Artemis I – the first launch under the agency's Artemis Program – will test SLS and Orion as an integrated system prior to crewed flights to the Moon.

**Final caption** (559 chars):
> A massive white building featuring a large painted American flag on its side and the NASA logo on its front face stands beneath a sky dominated by dark, heavy storm clouds. In the foreground, a large digital display screen housed within a blue structure shows a rover on a red, rocky surface with the text "MARS 2020" printed in the lower left corner. An American flag flies from a tall pole next to the display. Green grass covers the foreground field, extending toward the base of the building where smaller structures and trees are visible in the distance.

---

### KSC-97pc573  

<img src="images_thumb/KSC-97pc573.jpg" width="420"/>

**Seed (cleaned):**
> Like a rising sun lighting up the afternoon sky, the Space Shuttle Columbia soars from Launch Pad 39A at 2:20:32 p.m. EST, April 4, on the 16-day Microgravity Science Laboratory-1 (MSL-1) mission. The crew members are Mission Commander James D. Halsell, Jr.; Pilot Susan L. Still; Payload Commander Janice Voss; Mission Specialists Michael L. Gernhardt and Donald A. Thomas; and Payload Specialists Roger K. Crouch and Gregory T. Linteris.

**Final caption** (386 chars):
> A space shuttle stack ascends vertically into a deep blue sky, generating a massive, billowing cloud of white exhaust smoke and bright orange flames at its base. Dark, twisted tree branches frame the foreground, their silhouetted forms creating a natural arch over the launch plume. Green vegetation is visible at the bottom right, and a bird flies in the upper right corner of the sky.

---

### KSC-99pp0879  

<img src="images_thumb/KSC-99pp0879.jpg" width="420"/>

**Seed (cleaned):**
> First Lady Hillary Rodham Clinton (right) and her daughter, Chelsea, are greeted by NASA Astronaut Scott E. Parazynski (left) upon their arrival at the Skid Strip at Cape Canaveral Air Station to view the launch of Space Shuttle mission STS-93. Liftoff is scheduled for 12:36 a.m. EDT July 20. Much attention has been generated over the launch due to Commander Eileen M. Collins, the first woman to serve as commander of a Shuttle mission.

**Final caption** (617 chars):
> A man in a blue flight suit stands in profile on the left, facing a group of people, wearing a circular patch on his right shoulder with the text "STS-114" visible at the bottom. He looks toward two women in the foreground who are smiling; the woman in the center wears a dark blazer over a light blue collared shirt and holds a small blue object, while the woman to her right, in a navy blazer, touches her chin. Behind them, a line of people includes other astronauts in blue flight suits and women in business attire, standing near a vertically hanging American flag and a set of illuminated stairs leading upward.

---

### LRC-1995-B701_P-05869  

<img src="images_thumb/LRC-1995-B701_P-05869.jpg" width="420"/>

**Seed (cleaned):**
> Gus Grissom trying on a Spacesuit; Seated with assistant; Seated with assistant putting on boots; Standing by mirror, name tag visible; Outside in suit, name tag visible. Mercury Project photo, 1961. Original negatives sent to Johnson Space Center when astronauts moved to that center.

**Final caption** (530 chars):
> A man in a plaid shirt kneels on a carpeted floor, adjusting the lower leg and boot area of a shiny, metallic spacesuit worn by a seated astronaut. The astronaut sits in a wooden chair with a cushion, looking down as the kneeling man manipulates straps and fasteners near the ankle. The pressure suit features a high collar, chest harness, and hoses running along the limbs, with a patch visible on the left upper arm. In the background, a person in a white shirt and tie stands near dark metal lockers, observing the preparation.

---

### MSFC-1800680  

<img src="images_thumb/MSFC-1800680.jpg" width="420"/>

**Seed (cleaned):**
> High school and university students competed in the 2018 Human Exploration Rover Challenge event at the U.S. Space and Rocket Center in Huntsville, Alabama. Students came from across the U.S. as well as several foreign countries such as Brazil, Germany, India, and Mexico. This event, which is normally a 2 day event, was shortened to 1 day in 2018 due to adverse weather conditions.

**Final caption** (535 chars):
> Two people wearing helmets with blue and white graphics ride a custom-built, blue-framed recumbent tricycle with large wheels wrapped in bicycle chains. The vehicle, marked with a white plate reading "22" and "MT," sits on a wooden ramp structure. The rider in the foreground wears a black long-sleeved shirt and pants, while the second rider behind wears a light blue shirt. In the background, a large, dark green aircraft fuselage with "UNITED STATES" lettering rests on the tarmac near orange safety cones and a group of spectators.

---

### MSFC-201900555  

<img src="images_thumb/MSFC-201900555.jpg" width="420"/>

**Seed (cleaned):**
> AI. SpaceFactory of New York and Pennsylvania State University of College Park print subscale habitat structures at NASA's 3D-Printed Habitat Challenge, held at the Caterpillar Edwards Demonstration & Learning Center in Edwards, Illinois, May 1-4, 2019. The habitat print is the final level of the multi-phase competition, which began in in 2015.

**Final caption** (629 chars):
> A large, circular wall structure made of layered gray material is being constructed in the foreground by a 3D printing process. An orange robotic arm with a square nozzle extends from the right, extruding the material in continuous horizontal layers to build the curved wall. To the left, stacks of green and white bags are piled on the floor. Behind a metal railing draped with red fabric, a group of people wearing white hard hats and dark blue work shirts observe the operation, with some looking at a laptop screen and others watching the machinery. Yellow construction equipment is visible in the background of the workshop.

---

### NHQ201810310008  

<img src="images_thumb/NHQ201810310008.jpg" width="420"/>

**Seed (cleaned):**
> NASA's Jet Propulsion Laboratory InSight deputy principal investigator Sue Smrekar gives remarks during a NASA InSight Mars Lander (Interior Exploration using Seismic Investigations, Geodesy and Heat Transport) media briefing, Wednesday, Oct. 31, 2018 at NASA Headquarters in Washington.

**Final caption** (384 chars):
> A woman with short brown hair stands in front of a projection screen, wearing a maroon button-down shirt featuring a patch on the right chest that reads "InSight EOL" and "LANDING TEAM" above a graphic of a lander on a reddish surface. She gestures with her left hand raised while holding a black remote in her right hand. A large blue NASA logo is projected on the screen behind her.

---

### S69-41359  

<img src="images_thumb/S69-41359.jpg" width="420"/>

**Seed (cleaned):**
> Astronauts Michael Collins (left) and Edwin E. Aldrin Jr., are greeted by Dr. Robert R. Gilruth, director, Manned Spacecraft Center (MSC), and others upon their release from quarantine. The Apollo 11 crew left the Crew Reception Area (CRA) of the Lunar Receiving Laboratory (LRL) at 9 p.m., Aug. 10, 1969.

**Final caption** (479 chars):
> A man with a mustache, wearing a short-sleeved shirt with a visible identification badge on the left chest, extends his hand to shake hands with a balding man in a dark suit. Behind the man in the shirt, another man in a light-colored shirt with a lanyard around his neck looks on. To the right, a man wearing glasses holds a handheld radio transceiver with an antenna. The group stands in a narrow corridor with a drop ceiling and fluorescent lighting fixtures visible overhead.

---

### iss043e198394  

<img src="images_thumb/iss043e198394.jpg" width="420"/>

**Seed (cleaned):**
> Photographic documentation taken prior to collection of surface and air samples using various devices in multiple locations to characterize the types of microbial populations on the International Space Station (ISS) for the Microbial Observatory-1 payload. The Microbial Payload Tracking Series (Microbial Observatory-1) investigation monitors the types of microbes present on ISS over a one-year period. Samples are returned to Earth for further study.

**Final caption** (545 chars):
> Two black vertical modules, each topped with a handwritten white sign reading "ARED FLYWHEEL COVER," are mounted on a blue and silver rack system. A blue cylindrical component is attached to the right side of the central assembly. To the left, a yellow sticky note with "PLEASE" written at the top is affixed to a white panel. Small labels on the black modules display a green triangle symbol and the text "DO NOT REMOVE." The surrounding area includes white walls, exposed wiring, and a white storage bag, consistent with a spacecraft interior.

---
