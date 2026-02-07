#include "pogobot.h"
#include "time.h"

#define INFRARED_POWER 2 // 1,2,3

#define FQCY 60 // control update frequency. 30Hz | 60 Hz | 90 Hz | etc.
#define MAX_NB_OF_MSG 3 // max. number of messages per step which this robot can record // 3

#define SEND_MODE_ALLDIRECTION true // true: all direction at once; false: 4x one-direction
#define MSG_MODE_FULL_HEADER true // true: full header; false: short header

#define DEBUG_LEVEL 0 // 0: nothing; 1: debug; 2: synchronzation; 3: communication

#define BOOT_TIME 5 // waiting time before the start of the experience in seconds
#define LIGHT_THRESHOLD 10

#define RT_RUN_MIN_MS             1500                             // RUN = se déplacer en ligne droite (ou arrière) pendant 1.5 à 3 s.
#define RT_RUN_MAX_MS             3000
#define RT_TUMBLE_MIN_MS           400                             // TUMBLE = tourner sur place pendant 0.4 à 0.6 s.
#define RT_TUMBLE_MAX_MS           600

// Sous-phases pour le run&tumble (utile pour alterner).
typedef enum { RT_PHASE_RUN=0, RT_PHASE_TUMBLE } rt_phase_t;

// ===================== État global =====================
// Grosse structure "g" = tous les états du robot au runtime. "static userdata_t g;" = unique, global à ce fichier.
typedef struct {
    // Calibration moteurs
    uint8_t dirL, dirR;                                            // Sens logique (0/1) des moteurs gauche/droite. Exemple : 0=avant,1=arrière.
    uint32_t pwmLt, pwmRt, pwmLr, pwmRr;

    // Mode
    mobile_mode_t mode;                                            // État courant de la FSM (NORMAL/ALIGN/CONNECTED).

    // Run & tumble
    rt_phase_t rt_phase;                                           // RUN ou TUMBLE.
    ptimer_t   rt_phase_timer;                                     // Timer pour savoir quand basculer.
    uint32_t   rt_run_dur_ms;                                      // Durée actuelle du RUN (tirée au sort).
    uint32_t   rt_tumble_dur_ms;                                   // Durée actuelle du TUMBLE (tirée au sort).
    bool       rt_tumble_left;                                     // Pendant TUMBLE, tourne à gauche ? (Sinon à droite).
    bool       rt_run_backward;      // nouveau                      // Pendant RUN, va en marche arrière ? (sinon avant).

    // IR / contact global
    ptimer_t   contact_timer;                                      // Dernier contact IR (utile si tu veux mesurer "depuis quand").

    // Ancre courante
    grid_msg_t anchor_gm;          // si POL_GRID => (row,col) cible; si POL_AXIS0 => (row,col) voisin
    uint8_t    anchor_policy;      // POL_GRID ou POL_AXIS0         // Politique active pour l'ancrage actuel.
    ptimer_t   anchor_seen_timer;  // utilisé aussi pour l’hystérésis de retarget // Dernière fois où on a "vu" l'ancre.
    ptimer_t   anchor_first_seen;                                   // Utile pour l'UX (bleu pendant les X premières ms).

    // CONNECTED
    ptimer_t   connected_since;                                    // Depuis quand on est en CONNECTED (pour "vert latch").
    ptimer_t   orient_ok_hold;                                     // Maintien de "orientation OK".
    ptimer_t   misaligned_since;   // nouveau                       // Depuis quand on est "mal orienté" (pour quitter CONNECTED -> ALIGN).

    // Index local (émetteur)
    uint16_t myid;                                                 // ID matériel du robot (unique).
    bool     have_index;                                           // Connaît-on notre (row,col) ? (true/false).
    uint8_t  row;                                                  // Notre ligne courante.
    uint8_t  col;                                                  // Notre colonne courante.
    ptimer_t send_period;                                          // Timer pour cadence d'émission (gated).

    // UX (ventral)
    uint8_t  last_rx_face_led;     // 1..4                         // Quelle face (1..4) a reçu le dernier message récent.
    ptimer_t rx_face_seen;                                         // Depuis quand on a vu cette face (pour faire clignoter 1s).
    ptimer_t belly_blink;                                          // Timer de clignotement ON/OFF.
    bool     belly_on;                                             // État du clignotement (ON/OFF).

    // Cohérence RX
    ptimer_t coh_recent;                                           // Dernière fois où tout était cohérent (keepalive "fort").
    ptimer_t incoh_recent;                                         // Dernière fois où on a vu de l'incohérence (diagnostic/extension).
    uint8_t  coherent_streak;                                      // Combien de "coups" cohérents d'affilée (≥2 pour CONNECTED).
    uint8_t  incoherent_streak;                                    // Combien d'incohérents d'affilée.

    // Cumul E/S pour l’ancrage
    bool     anchor_has_E;                                         // A-t-on vu "E" récemment (voisin émettant côté Est) ?
    bool     anchor_has_S;                                         // A-t-on vu "S" récemment (voisin émettant côté Sud) ?
    ptimer_t anchor_E_seen;                                        // Timer de fraîcheur "E".
    ptimer_t anchor_S_seen;                                        // Timer de fraîcheur "S".

    // --------- Ajouts anti-coinçage ---------
    ptimer_t align_since;               // Chrono d'entrée en ALIGN (pour ALIGN_MAX_MS)
    ptimer_t anchor_reentry_block;      // Horloge d'immunité après sortie forcée d'ALIGN
    bool     anchor_reentry_block_active; // Flag immunité active

} userdata_t;

static userdata_t g; // état global                                 // Déclare l'unique instance globale "g" (initialisée à 0 par défaut).

// ===================== Moteurs =====================
// Les fonctions ci-dessous masquent le détail bas niveau (sens + puissance) pour exprimer des intentions : avancer, reculer, tourner, etc.
static inline void motors_forward(void){
    pogobot_motor_dir_set(motorL, g.dirL);                          // Fixe la direction "avant logique" du moteur gauche.
    pogobot_motor_dir_set(motorR, g.dirR);                          // Fixe la direction "avant logique" du moteur droit.
    pogobot_motor_power_set(motorL, g.pwmLr);                        // Envoie la puissance prédéfinie à gauche.
    pogobot_motor_power_set(motorR, g.pwmRr);                        // Envoie la puissance prédéfinie à droite.
}
static inline void motors_backward(void){
    pogobot_motor_dir_set(motorL, 1 - g.dirL);                      // Inverse la direction gauche (1-dirL = "arrière logique").
    pogobot_motor_dir_set(motorR, 1 - g.dirR);                      // Inverse la direction droite.
    pogobot_motor_power_set(motorL, g.pwmLr);                        // Même puissances qu'en avant (symétrie).
    pogobot_motor_power_set(motorR, g.pwmRr);
}
static inline void motors_stop(void){
    pogobot_motor_power_set(motorL, 0);                             // Met puissance à 0 → arrêt moteur gauche.
    pogobot_motor_power_set(motorR, 0);                             // Idem à droite.
}
// Rotation GAUCHE sur place (moteurs inverses)
static inline void motors_turn_left_in_place(void){
    pogobot_motor_dir_set(motorL, 1 - g.dirL); // gauche en arrière // On fait reculer la gauche...
    pogobot_motor_dir_set(motorR,     g.dirR); // droite en avant   // ...et avancer la droite → pivot sur place vers la gauche.
    pogobot_motor_power_set(motorL, g.pwmLt);               // Utilise puissances spécifiques "rotation gauche".
    pogobot_motor_power_set(motorR, g.pwmRt);
}
// Rotation DROITE sur place (miroir)
static inline void motors_turn_right_in_place(void){
    pogobot_motor_dir_set(motorL,     g.dirL); // gauche en avant   // Avance la gauche...
    pogobot_motor_dir_set(motorR, 1 - g.dirR); // droite en arrière // ...recule la droite → pivot sur place vers la droite.
    pogobot_motor_power_set(motorL, g.pwmLt);              // Utilise puissances "rotation droite".
    pogobot_motor_power_set(motorR, g.pwmRt);
}

static void app_init(void){
    srand(pogobot_helper_getRandSeed());                              // Initialise rand() avec une graine non triviale (SDK).
    pogobot_infrared_set_power(INFRARED_POWER);                       // Fixe la puissance IR selon notre réglage.

    uint8_t mem[3] = {0};                                             // Petit tampon (3 octets) pour récup calibration moteurs.
    pogobot_motor_dir_mem_get(mem);                                   // Lecture calibration (SDK). Convention : mem[0]=droite, mem[1]=gauche.
    g.dirR = mem[0];                                                  // Applique la direction logique du moteur droit.
    g.dirL = mem[1];                                                  // Applique la direction logique du moteur gauche.

    // Puissances récupérées depuis EEPROM (SDK)
    uint16_t pwr[3] = {0};
    if (pogobot_motor_power_mem_get(pwr) == 0) {
        g.pwmRt = pwr[0] * FACT1;
        g.pwmLt = pwr[1] * FACT1;
        g.pwmRr = pwr[0] * FACT;
        g.pwmLr = pwr[1] * FACT;


        printf("Puissances R : L=%u, R=%u\n", g.pwmLr, g.pwmRr);
        printf("Puissances T : L=%u, R=%u\n", g.pwmLt, g.pwmRt);
    } else {
        printf("Erreur de lecture de la mémoire moteur !\n");
    }

    g.mode = MODE_NORMAL;                                             // On démarre en exploration.

    // Run & tumble : on tire les premières durées et directions.
    g.rt_phase = RT_PHASE_RUN;                                        // Première phase = RUN (avancer/reculer).
    ptimer_start(&g.rt_phase_timer);                                  // Timer de phase RT démarré maintenant.
    g.rt_run_dur_ms    = rand_between(RT_RUN_MIN_MS, RT_RUN_MAX_MS);  // RUN durera entre 1.5 et 3 s.
    g.rt_tumble_dur_ms = rand_between(RT_TUMBLE_MIN_MS, RT_TUMBLE_MAX_MS); // TUMBLE durera 0.4..0.6 s.
    g.rt_tumble_left   = (rand() & 1) != 0;                           // TUMBLE initial : gauche si bit aléatoire=1 (sinon droite).
    g.rt_run_backward  = (rand() & 1) != 0;                           // RUN initial : arrière si bit=1 (sinon avant).

    // Ancre/cible : rien au départ, politique par défaut = AXIS0 (on accepte volontiers de s'accrocher à un bord).
    ptimer_start(&g.contact_timer);                                   // Reset du "dernier contact IR".
    g.anchor_gm.row = g.anchor_gm.col = 255;                          // Aucune ancre/cible.
    g.anchor_policy = POL_AXIS0; // par défaut                        // Politique de fallback : bords.
    ptimer_start(&g.anchor_seen_timer);                               // On commence à "pas avoir vu" l'ancre.
    ptimer_start(&g.anchor_first_seen);                               // Servira pour l'UX : bleu au début.

    // CONNECTED : timers prêts (même si on n'y est pas encore).
    ptimer_start(&g.connected_since);
    ptimer_start(&g.orient_ok_hold);
    ptimer_start(&g.misaligned_since);

    // Émetteur (id, timers UX) :
    emitter_init_state();

    // Cohérence : on part de 0.
    ptimer_start(&g.coh_recent);
    ptimer_start(&g.incoh_recent);
    g.coherent_streak   = 0;
    g.incoherent_streak = 0;

    // Requis E/S : rien vu pour l'instant.
    g.anchor_has_E = false; g.anchor_has_S = false;
    ptimer_start(&g.anchor_E_seen);
    ptimer_start(&g.anchor_S_seen);

    // Anti-coinçage : init des timers/flags
    ptimer_start(&g.align_since);
    ptimer_start(&g.anchor_reentry_block);
    g.anchor_reentry_block_active = false;

    // Feedback & moteurs : LED "run", et on démarre en avant/arrière selon le tirage.
    ui_head_normal_run();
    if (g.rt_run_backward) motors_backward(); else motors_forward();
}

static void update_run_tumble(void){                                   // Fonction appelée quand g.mode == MODE_NORMAL.
    if (g.rt_phase == RT_PHASE_RUN) {                                  // Si la sous-phase RT est RUN (translation).
        ui_head_normal_run();                                          // UX tête : couleur "run" (cyan).
        if (g.rt_run_backward) motors_backward();                      // Si le tirage aléatoire a choisi "reculer"...
        else                   motors_forward();                       // ...sinon on avance (mêmes puissances, sens inversé).

        if (ptimer_expired(&g.rt_phase_timer, g.rt_run_dur_ms)) {      // Si la durée RUN est écoulée, on bascule en TUMBLE.
            g.rt_phase = RT_PHASE_TUMBLE;                              // Change la sous-phase → TUMBLE (rotation sur place).
            ptimer_start(&g.rt_phase_timer);                           // Réarme le timer pour mesurer la durée du TUMBLE.
            g.rt_tumble_dur_ms = rand_between(RT_TUMBLE_MIN_MS, RT_TUMBLE_MAX_MS); // Tire une nouvelle durée TUMBLE aléatoire.
            g.rt_tumble_left   = (rand() & 1) != 0;   // tirage à chaque bascule // Tire au sort le sens de rotation (gauche/droite).
        }
    } else { // TUMBLE                                                  // Sous-phase RT : TUMBLE (rotation sur place).
        ui_head_normal_tumble();                                       // UX tête : couleur "tumble" (cyan assombri).
        if (g.rt_tumble_left) motors_turn_left_in_place();             // Si "gauche" → tourne vers la gauche sur place.
        else                  motors_turn_right_in_place();            // Sinon → tourne vers la droite sur place.

        if (ptimer_expired(&g.rt_phase_timer, g.rt_tumble_dur_ms)) {   // Si la durée TUMBLE est écoulée...
            g.rt_phase = RT_PHASE_RUN;                                 // ...on revient à RUN (translation).
            ptimer_start(&g.rt_phase_timer);                           // Réarme le timer pour la nouvelle durée RUN.
            g.rt_run_dur_ms   = rand_between(RT_RUN_MIN_MS, RT_RUN_MAX_MS); // Tire une nouvelle durée RUN.
            g.rt_run_backward = (rand() & 1) != 0;    // tirage à chaque bascule // Tire au sort "avant" vs "arrière".
            if (g.rt_run_backward) motors_backward(); else motors_forward();   // Applique immédiatement la direction choisie.
        }
    }
}

int main(void) {

    // init (mandatory)
    pogobot_init();
    srand(pogobot_helper_getRandSeed()); // initialize the random number generator
    pogobot_infrared_set_power(INFRARED_POWER); // set the power level used to send all the next messages
    
    if (DEBUG_LEVEL) {
        printf("\n");
        printf("[INFO] =-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
        printf("[INFO] =-=- POGOBOT::METADATA -=-=\n");
        printf("[INFO] =-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
        printf("[INFO] INFRARED_POWER         %d\n",INFRARED_POWER);
        printf("[INFO] FQCY                   %d\n",FQCY);
        printf("[INFO] MAX_NB_OF_MSG          %d\n",MAX_NB_OF_MSG);
        printf("[INFO] SEND_MODE_ALLDIRECTION %d\n",SEND_MODE_ALLDIRECTION);
        printf("[INFO] MSG_MODE_FULL_HEADER   %d\n",MSG_MODE_FULL_HEADER);
        printf("[INFO] DEBUG_LEVEL            %d\n",DEBUG_LEVEL);
        printf("[INFO] =-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    }


    // ********************************************************************************
    // * Initialization: hanabi experimental parameters and initial conditions
    // ********************************************************************************
    
    // ********************************************************************************
    // * Start-up phase for simultaneous start of the robots when the lights turn off
    // ********************************************************************************

    pogobot_led_setColor(red.r, red.g, red.b); // set initial led color to red

    if (DEBUG_LEVEL == 1) 
        printf("[I'm Pogobot %d] Photosensor msg_values 0: %d, 1: %d, 2: %d\n", my_pogobot_id, photo0, photo1, photo2);

    pogobot_led_setColor(white.r, white.g, white.b); // set boot led color to white
    sleep(BOOT_TIME);
    // pogobot_stopwatch_reset(&timeout_age_watch); // reset of the timer, for age timeout
    
    app_init();

    // ********************************************************************************
    // * Main loop
    // ********************************************************************************

    while (1)
    {
        pogobot_stopwatch_reset(&mystopwatch); // reset of the timer, for step synchronization

        


        // ********************************************************************************
        // * Reinitialize the age counter in case of stagnation
        // ********************************************************************************

        // seconds = pogobot_stopwatch_get_elapsed_microseconds(&timeout_age_watch) / 1000000; // 1 s = 1000000 microseconds
        // if (seconds >= timeout_age_s) {
        //     if (DEBUG_LEVEL == 1)
        //         printf("[I'm Pogobot %d] [RESET] The age hasn't changed since %lu seconds. Reinitialization of HANABI: age = 0\n", my_pogobot_id, seconds);

        //     age = 0;
        //     led1_status = 0;
        //     pogobot_led_setColors(0, 0, 0, 1); // led 1 off
        //     pogobot_led_setColor(white.r, white.g, white.b); // set upper led color to white
        //     sleep(timeout_age_s);
        //     pogobot_stopwatch_reset(&timeout_age_watch); // reset of the timer, for age timeout
        // }
        

        // ********************************************************************************
        // * Step synchronize: wait for next step (if not timed out already)
        // ********************************************************************************

        microseconds = pogobot_stopwatch_get_elapsed_microseconds(&mystopwatch);

        if (microseconds < 1000000 / FQCY) { // 1 s = 1000000 microseconds
            if (DEBUG_LEVEL == 2) {
                printf("[TIME] Step took %lu usec. Sleep until next tick.\n", microseconds);
                pogobot_led_setColors(0, 25, 0, 4);
            }
            msleep(((1000000 / FQCY) - microseconds) / 1000); // wait for next step. NB: msleep in ms
        }
        else {
            if (DEBUG_LEVEL == 2) {
                printf("[TIME] Step took %lu usec, should be less than %u usec. [ ### TIME OVERFLOW ### ]\n", microseconds, (1000000 / FQCY));
                pogobot_led_setColors(25, 25, 25, 4); // too slow. Continue directly to next step
            }
        }
    }
}
