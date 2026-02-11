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

#define MOTOR_POWER 718 // default power of the motor in the else case
#define IDENTIFIER 99
#define P_MSG 1

// ********************************************************************************
// * Initialization: Pogobot led colors structure
// ********************************************************************************

typedef struct {
    char name[16];
    uint8_t r;
    uint8_t g;
    uint8_t b;
} rgb_color; 

rgb_color red =         {.name = "red",        .r = 25, .g = 0,  .b = 0};
rgb_color green =       {.name = "green",      .r = 0,  .g = 25, .b = 0};
rgb_color blue =        {.name = "blue",       .r = 0,  .g = 0,  .b = 25};
rgb_color magenta =     {.name = "magenta",    .r = 25, .g = 0,  .b = 25};
rgb_color yellow =      {.name = "yellow",     .r = 12, .g = 6,  .b = 0};
rgb_color cyan =        {.name = "cyan",       .r = 0,  .g = 25, .b = 25};
rgb_color orange =      {.name = "orange",     .r = 25, .g = 6,  .b = 0};
rgb_color purple =      {.name = "purple",     .r = 6,  .g = 0,  .b = 25};
rgb_color light_pink =  {.name = "light_pink", .r = 12, .g = 3,  .b = 12};
rgb_color mint_green =  {.name = "mint_green", .r = 6,  .g = 25, .b = 6};
rgb_color white =       {.name = "white",      .r = 25, .g = 25, .b = 25};

#define RT_RUN_MIN_MS             1500                             // RUN = se déplacer en ligne droite (ou arrière) pendant 1.5 à 3 s.
#define RT_RUN_MAX_MS             3000
#define RT_TUMBLE_MIN_MS           400                             // TUMBLE = tourner sur place pendant 0.4 à 0.6 s.
#define RT_TUMBLE_MAX_MS           600

#define FACT 1.0
#define FACT1 1.0

typedef enum { 
    MODE_NORMAL, 
    MODE_ALIGN, 
    MODE_CONNECTED,
    MODE_ROOT
} mobile_mode_t;

// Sous-phases pour le run&tumble (utile pour alterner).
typedef enum { RT_PHASE_RUN=0, RT_PHASE_TUMBLE } rt_phase_t;

// ===================== État global =====================

uint8_t dirL, dirR;
uint32_t pwmLt, pwmRt, pwmLr, pwmRr;

mobile_mode_t mode;        
rt_phase_t rt_phase;
time_reference_t rt_phase_timer; 

uint32_t rt_duration_ms;   
bool rt_tumble_left;
bool rt_run_backward;      
int robot_identifier  = IDENTIFIER;

// ===================== Fonctions Utiles =====================
static uint32_t rand_between(uint32_t min, uint32_t max) {
    return min + (rand() % (max - min + 1));
}
// ===================== Moteurs =====================
// Les fonctions ci-dessous masquent le détail bas niveau (sens + puissance) pour exprimer des intentions : avancer, reculer, tourner, etc.

static inline void motors_stop(void){
    pogobot_motor_power_set(motorL, 0);                             // Met puissance à 0 → arrêt moteur gauche.
    pogobot_motor_power_set(motorR, 0);                             // Idem à droite.
}
static void motors_forward(void) {
    pogobot_motor_dir_set(motorL, dirL);
    pogobot_motor_dir_set(motorR, dirR);
    pogobot_motor_power_set(motorL, pwmLr);
    pogobot_motor_power_set(motorR, pwmRr);
}

static void motors_backward(void) {
    pogobot_motor_dir_set(motorL, 1 - dirL);
    pogobot_motor_dir_set(motorR, 1 - dirR);
    pogobot_motor_power_set(motorL, pwmLr);
    pogobot_motor_power_set(motorR, pwmRr);
}

static void motors_turn_left(void) {
    pogobot_motor_dir_set(motorL, 1 - dirL);
    pogobot_motor_dir_set(motorR, dirR);
    pogobot_motor_power_set(motorL, pwmLt);
    pogobot_motor_power_set(motorR, pwmRt);
}

static void motors_turn_right(void) {
    pogobot_motor_dir_set(motorL, dirL);
    pogobot_motor_dir_set(motorR, 1 - dirR);
    pogobot_motor_power_set(motorL, pwmLt);
    pogobot_motor_power_set(motorR, pwmRt);
}

static void app_init(void){
    pogobot_infrared_set_power(INFRARED_POWER);                       // Fixe la puissance IR selon notre réglage.
    if(robot_identifier==0) mode = MODE_ROOT;
    else{
        uint8_t mem[3] = {0};                                             // Petit tampon (3 octets) pour récup calibration moteurs.
        pogobot_motor_dir_mem_get(mem);                                   // Lecture calibration (SDK). Convention : mem[0]=droite, mem[1]=gauche.
        dirR = mem[0];                                                  // Applique la direction logique du moteur droit.
        dirL = mem[1];                                                  // Applique la direction logique du moteur gauche.

        
        uint16_t pwr[3] = {0};
        if (pogobot_motor_power_mem_get(pwr) == 0) {
            pwmRt = pwr[0] * FACT1;
            pwmLt = pwr[1] * FACT1;
            pwmRr = pwr[0] * FACT;
            pwmLr = pwr[1] * FACT;


            printf("Puissances R : L=%lu, R=%lu\n", (unsigned long)pwmLr, (unsigned long)pwmRr);
            printf("Puissances T : L=%lu, R=%lu\n", (unsigned long)pwmLt, (unsigned long)pwmRt);
        } else {
            printf("Erreur de lecture de la mémoire moteur ! On utilise des valeurs par défault\n");
            pwmRt = MOTOR_POWER * FACT1; 
            pwmLt = MOTOR_POWER * FACT1;
            pwmRr = MOTOR_POWER * FACT;
            pwmLr = MOTOR_POWER * FACT;
            
        }

        mode = MODE_NORMAL;                                             // On démarre en exploration.

        // Run & tumble : on tire les premières durées et directions.
        rt_phase = RT_PHASE_RUN;                                        // Première phase = RUN (avancer/reculer).
        pogobot_stopwatch_reset(&rt_phase_timer);                               // Timer de phase RT démarré maintenant.
        rt_duration_ms = rand_between(RT_RUN_MIN_MS, RT_RUN_MAX_MS);
        rt_tumble_left   = rand() % 2;                           // TUMBLE initial : gauche si bit aléatoire=1 (sinon droite).
        rt_run_backward  = rand() % 2;                           // RUN initial : arrière si bit=1 (sinon avant).

        

        // Feedback & moteurs : LED "run", et on démarre en avant/arrière selon le tirage.
        if (rt_run_backward) motors_backward(); else motors_forward();
    }
}

static void update_run_tumble(void) {
    // Calcul du temps écoulé en ms
    uint32_t elapsed = (uint32_t)(pogobot_stopwatch_get_elapsed_microseconds(&rt_phase_timer) / 1000);

    if (rt_phase == RT_PHASE_RUN) {
        //pogobot_led_setColor(cyan.r, cyan.g, cyan.b);
        if (rt_run_backward) motors_backward();                      // Si le tirage aléatoire a choisi "reculer"...
        else                   motors_forward();

        if (elapsed >= rt_duration_ms) {
            rt_phase = RT_PHASE_TUMBLE;
            pogobot_stopwatch_reset(&rt_phase_timer);
            rt_duration_ms = rand_between(RT_TUMBLE_MIN_MS, RT_TUMBLE_MAX_MS);
            rt_tumble_left = rand() % 2;
        }
    } else {
        //pogobot_led_setColor(yellow.r, yellow.g, yellow.b);
        if (rt_tumble_left) motors_turn_left(); else motors_turn_right();

        if (elapsed >= rt_duration_ms) {
            rt_phase = RT_PHASE_RUN;
            pogobot_stopwatch_reset(&rt_phase_timer);
            rt_duration_ms = rand_between(RT_RUN_MIN_MS, RT_RUN_MAX_MS);
            rt_run_backward = rand() % 2; 
        }
    }
}

static void update_root(void){
    pogobot_led_setColor(green.r, green.g, green.b);
    if(rand()*100<=P_MSG){
        pogobot_led_setColors(red.r, red.g, red.b, 3);
        pogobot_infrared_sendLongMessage_uniSpe(2, (__uint8_t*)4, (__uint16_t) 4);

    }
}

static void update_mode(void){
    if(mode!= MODE_ROOT){
        pogobot_infrared_update();
        if (pogobot_infrared_message_available()) {
            for(int i=0;i<5;i++) pogobot_led_setColors(0,0,0,i);
            //pogobot_led_setColor(purple.r, purple.g, purple.b);
            int msg_rcv = 0;
            mode = MODE_CONNECTED;
            while (pogobot_infrared_message_available() && msg_rcv < MAX_NB_OF_MSG){
                message_t mr;
                pogobot_infrared_recover_next_message(&mr);
                uint8_t ir = mr.header._receiver_ir_index + 1;
                pogobot_led_setColors(purple.r, purple.g, purple.b, ir);
                msg_rcv++;
            }
        }
    }
    switch(mode){
        case MODE_NORMAL:
            update_run_tumble();
        case MODE_ROOT:
            update_root();
        case MODE_ALIGN:
            return;
        case MODE_CONNECTED:
            motors_stop();
    }
}

int main(void) {

    // init (mandatory)
    pogobot_init();
    srand(pogobot_helper_getRandSeed()); // initialize the random number generator
    //pogobot_infrared_set_power(INFRARED_POWER); // set the power level used to send all the next messages
    
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
    time_reference_t mystopwatch; // timer for step synchronization 
    uint32_t microseconds = 0; // counter for step synchronization
    // ********************************************************************************
    // * Start-up phase for simultaneous start of the robots when the lights turn off
    // ********************************************************************************

    pogobot_led_setColor(red.r, red.g, red.b); // set initial led color to red

    if (DEBUG_LEVEL == 1) 
        //printf("[I'm Pogobot %d] Photosensor msg_values 0: %d, 1: %d, 2: %d\n", my_pogobot_id, photo0, photo1, photo2);

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
        
        update_mode();
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
