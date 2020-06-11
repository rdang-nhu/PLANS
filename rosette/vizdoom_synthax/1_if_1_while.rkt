(define-synthax Control_Flow_Block
  ([(Control_Flow_Block previous_actions perceptions)
      (choose
       (If previous_actions perceptions)
       (While previous_actions perceptions))]))

(define-synthax Block
  ([(Block previous_actions perceptions)
     (let ([new_actions (Action_Block previous_actions 3)])
            (let ([new_actions (Control_Flow_Block new_actions perceptions)])
                (let ([new_actions (Action_Block new_actions 3)])
                    (let ([new_actions (Control_Flow_Block new_actions perceptions)])
                          (Action_Block new_actions 3)))))]))